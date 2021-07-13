/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include "server_ucc.h"
#include "host_channel.h"
#include "ucc/api/ucc.h"

#define CORES 8
#define MAX_THREADS 128
typedef struct {
    pthread_t id;
    int idx, nthreads;
    dpu_ucc_comm_t comm;
    dpu_hc_t *hc;
    unsigned int itt;
} thread_ctx_t;

/* thread accessible data - split reader/writer */
typedef struct {
    volatile unsigned long todo;    /* first cache line */
    volatile unsigned long pad1[3]; /* pad to 64bytes */
    volatile unsigned long done;    /* second cache line */
    volatile unsigned long pad2[3]; /* pad to 64 bytes */
} thread_sync_t;

static thread_sync_t *thread_sync = NULL;

static void dpu_thread_set_affinity(thread_ctx_t *ctx)
{
    int i = 0;
    int places = CORES/ctx->nthreads;
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    
    CPU_ZERO(&cpuset);
	for (i = 0; i < places; i++) {
		CPU_SET((ctx->idx*places)+i, &cpuset);
	}
    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

static void dpu_wait_for_work(thread_ctx_t *ctx)
{
    int i;
    if (!ctx->idx) {
        ctx->itt++;
        dpu_hc_wait(ctx->hc, ctx->itt);
        for(i=0; i<ctx->nthreads; i++) {
            thread_sync[i].done = 0;
            thread_sync[i].todo = 1;
        }
    }
    while(!thread_sync[ctx->idx].todo);
}

static void dpu_mark_work_done(thread_ctx_t *ctx)
{
    int i;
    thread_sync[ctx->idx].todo = 0;
    thread_sync[ctx->idx].done = 1;
    if (!ctx->idx) {
        for(i=0; i<ctx->nthreads; i++) {
            while(!thread_sync[i].done);
        }
        dpu_hc_reply(ctx->hc, ctx->itt);
    }
}

void dpu_coll_init_allreduce(thread_ctx_t *ctx, ucc_coll_req_h *request)
{
    size_t count = dpu_hc_get_count_total(ctx->hc);
    size_t dt_size = dpu_ucc_dt_size(dpu_hc_get_dtype(ctx->hc));
    size_t block = count / ctx->nthreads;
    size_t offset = block * ctx->idx;

    if(ctx->idx < (count % ctx->nthreads)) {
        offset += ctx->idx;
        block++;
    } else {
        offset += (count % ctx->nthreads);
    }
    
    ucc_coll_args_t coll_args = {
        .coll_type = UCC_COLL_TYPE_ALLREDUCE,
        .mask      = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS,
        .src.info = {
            .buffer   = ctx->hc->mem_segs.put.base + offset * dt_size,
            .count    = block,
            .datatype = dpu_hc_get_dtype(ctx->hc),
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .dst.info = {
            .buffer     = ctx->hc->mem_segs.get.base + offset * dt_size,
            .count      = block,
            .datatype   = dpu_hc_get_dtype(ctx->hc),
            .mem_type   = UCC_MEMORY_TYPE_HOST,
        },
        .reduce = {
            .predefined_op = dpu_hc_get_op(ctx->hc),
        }
    };

    UCC_CHECK(ucc_collective_init(&coll_args, request, ctx->comm.team));
}

void dpu_coll_init_alltoall(thread_ctx_t *ctx, ucc_coll_req_h *request)
{
    /* Multithreading not supported */
    if(ctx->idx > 0) {
        *request = NULL;
        return;
    }

    size_t count = dpu_hc_get_count_total(ctx->hc);
    size_t dt_size = dpu_ucc_dt_size(dpu_hc_get_dtype(ctx->hc));
    size_t block = count;
    size_t offset = 0;
    
    ucc_coll_args_t coll_args = {
        .coll_type = UCC_COLL_TYPE_ALLTOALL,
        .src.info = {
            .buffer   = ctx->hc->mem_segs.put.base + offset * dt_size,
            .count    = block,
            .datatype = dpu_hc_get_dtype(ctx->hc),
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .dst.info = {
            .buffer     = ctx->hc->mem_segs.get.base + offset * dt_size,
            .count      = block,
            .datatype   = dpu_hc_get_dtype(ctx->hc),
            .mem_type   = UCC_MEMORY_TYPE_HOST,
        },
    };

    UCC_CHECK(ucc_collective_init(&coll_args, request, ctx->comm.team));
}

void *dpu_worker(void *arg)
{
    thread_ctx_t *ctx = (thread_ctx_t*)arg;
    ucc_coll_req_h request;

    dpu_thread_set_affinity(ctx);

    while(1) {
        dpu_wait_for_work(ctx);
        
        ucc_coll_type_t coll_type = dpu_hc_get_coll_type(ctx->hc);
        //fprintf(stderr, "Requested coll type: %d\n", coll_type);
        if (coll_type == UCC_COLL_TYPE_ALLREDUCE) {
            dpu_coll_init_allreduce(ctx, &request);
        } else if (coll_type == UCC_COLL_TYPE_ALLTOALL) {
            dpu_coll_init_alltoall(ctx, &request);
        } else if (coll_type == UCC_COLL_TYPE_LAST) {
            fprintf(stderr, "Received hangup, exiting loop\n");
            break;
        } else {
            fprintf(stderr, "Unsupported coll type: %d\n", coll_type);
            break;
        }

        if (request != NULL) {
            UCC_CHECK(ucc_collective_post(request));
            while (UCC_OK != ucc_collective_test(request)) {
                ucc_context_progress(ctx->comm.ctx);
            }
            UCC_CHECK(ucc_collective_finalize(request));
        }

        dpu_mark_work_done(ctx);
    }

    return NULL;
}

int main(int argc, char **argv)
{
//     fprintf (stderr, "%s\n", __FUNCTION__);
//     sleep(20);

    int nthreads = 0, i;
    thread_ctx_t *tctx_pool = NULL;
    dpu_ucc_global_t ucc_glob;
    dpu_hc_t hc_b, *hc = &hc_b;

    if (argc < 2 ) {
        printf("Need thread # as an argument\n");
        return 1;
    }
    nthreads = atoi(argv[1]);
    if (MAX_THREADS < nthreads || 0 >= nthreads) {
        printf("ERROR: bad thread #: %d\n", nthreads);
        return 1;
    }
    printf("DPU daemon: Running with %d threads\n", nthreads);
    tctx_pool = calloc(nthreads, sizeof(*tctx_pool));
    UCC_CHECK(dpu_ucc_init(argc, argv, &ucc_glob));

//     thread_sync = calloc(nthreads, sizeof(*thread_sync));
    thread_sync = aligned_alloc(64, nthreads * sizeof(*thread_sync));
    memset(thread_sync, 0, nthreads * sizeof(*thread_sync));

    dpu_hc_init(hc);
    dpu_hc_accept(hc);

    for(i = 0; i < nthreads; i++) {
//         printf("Thread %d spawned!\n", i);
        UCC_CHECK(dpu_ucc_alloc_team(&ucc_glob, &tctx_pool[i].comm));
        tctx_pool[i].idx = i;
        tctx_pool[i].nthreads = nthreads;
        tctx_pool[i].hc       = hc;
        tctx_pool[i].itt = 0;

        if (i < nthreads - 1) {
            pthread_create(&tctx_pool[i].id, NULL, dpu_worker,
                           (void*)&tctx_pool[i]);
        }
    }

    /* The final DPU worker is executed in this context */
    dpu_worker((void*)&tctx_pool[i-1]);

    for(i = 0; i < nthreads; i++) {
        if (i < nthreads - 1) {
            pthread_join(tctx_pool[i].id, NULL);
        }
        dpu_ucc_free_team(&ucc_glob, &tctx_pool[i].comm);
//         printf("Thread %d joined!\n", i);
    }

    dpu_ucc_finalize(&ucc_glob);
    return 0;
}
