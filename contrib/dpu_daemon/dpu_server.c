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
    dpu_get_sync_t ar_sync;
} thread_ctx_t;

/* thread accisble data - split reader/writer */
typedef struct {
    volatile unsigned int g_coll_id;  /* first cache line */
    volatile unsigned int pad[7]; /* pad to 64bytes */
    volatile unsigned int l_coll_id;  /* second cache line */
    volatile unsigned int pad2[7]; /* pad to 64 bytes */
} thread_sync_t;

static thread_sync_t *thread_main_sync = NULL;
static thread_sync_t *thread_sub_sync = NULL;
dpu_put_sync_t tmp_sync;

void *dpu_worker(void *arg)
{
    // fprintf (stdout, "sleeping %d\n", getpid());
    // sleep(20);
    thread_ctx_t *ctx = (thread_ctx_t*)arg;
    int places = CORES/ctx->nthreads;
    int i = 0, j = 0, inprogress = 0;
    dpu_put_sync_t *lsync = ctx->hc->mem_segs.sync.base;

    ucc_coll_req_h request;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    CPU_ZERO(&cpuset);
    
	for (i = 0; i < places; i++) {
		CPU_SET((ctx->idx*places)+i, &cpuset);
	}
    
    i = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);

    while(1) {
        /* Wait for operation to start */
        ctx->ar_sync.coll_id++;
        ctx->ar_sync.count_serviced = 0;
        if (ctx->idx > 0) {
            while (thread_main_sync[ctx->idx].g_coll_id < ctx->ar_sync.coll_id) {
                /* busy wait */
            }
        }
        else {
            /* Data arrived, main thread will synchronize sub threads */
            dpu_hc_wait(ctx->hc, ctx->ar_sync.coll_id);
            for (i = 0; i < ctx->nthreads; i++) {
                thread_main_sync[i].g_coll_id++;
            }
        }

        /* Hang up? */
        if (lsync->op    == UCC_OP_USERDEFINED &&
            lsync->dtype == UCC_DT_USERDEFINED) {
            break;
        }

        /* Process all data */
        do {
            thread_sub_sync[ctx->idx].g_coll_id++;
            if (ctx->idx > 0) { /* sub threads */
                while (thread_sub_sync[ctx->idx].l_coll_id < thread_sub_sync[ctx->idx].g_coll_id) {
                    /* busy wait */
                }
            }
            else { /* main */
                inprogress = 0;

                /* main waits for count_in to be updated from host */
                while(lsync->count_in <= ctx->ar_sync.count_serviced) {
                    /* busy wait */
                }

                /* Main thread syncs incoming data */
                memcpy(&tmp_sync, lsync, sizeof(dpu_put_sync_t));

                inprogress = 1;
                while (inprogress) {
                    inprogress = 0;
                    /* check if all threads are ready */
                    for (i = 1; i < ctx->nthreads; i++) {
                        if(thread_sub_sync[i].g_coll_id < thread_sub_sync[ctx->idx].g_coll_id ) {
                            inprogress++;
                            break;
                        }
                    }
                    if (!inprogress) {
                        /* Release sub threads */
                        for (i = 1; i < ctx->nthreads; i++) {
                            thread_sub_sync[i].l_coll_id++;
                        }
                        break;
                    }
                }
            }

            int offset, block;
            int count = tmp_sync.count_in - ctx->ar_sync.count_serviced;
            int ready = 0;
            int dt_size = dpu_ucc_dt_size(tmp_sync.dtype);

            block = count / ctx->nthreads;
            offset = ctx->ar_sync.count_serviced + block * ctx->idx;
            if(ctx->idx < (count % ctx->nthreads)) {
                offset += ctx->idx;
                block++;
            } else {
                offset += (count % ctx->nthreads);
            }
            
            ucc_coll_args_t coll = {
                .mask      = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS,
                .coll_type = UCC_COLL_TYPE_ALLREDUCE,
                .src.info = {
                    .buffer   = ctx->hc->mem_segs.put.base + offset * dt_size,
                    .count    = block * dt_size,
                    .datatype = tmp_sync.dtype,
                    .mem_type = UCC_MEMORY_TYPE_HOST,
                },
                .dst.info = {
                    .buffer     = ctx->hc->mem_segs.get.base + offset * dt_size,
                    .count      = block * dt_size,
                    .datatype   = tmp_sync.dtype,
                    .mem_type = UCC_MEMORY_TYPE_HOST,
                },
                .reduce = {
                    .predefined_op = tmp_sync.op,
                },
            };

            UCC_CHECK(ucc_collective_init(&coll, &request, ctx->comm.team));
            UCC_CHECK(ucc_collective_post(request));
            while (UCC_OK != ucc_collective_test(request)) {
                ucc_context_progress(ctx->comm.ctx);
            }
            UCC_CHECK(ucc_collective_finalize(request));

            thread_sub_sync[ctx->idx].g_coll_id++;
            ctx->ar_sync.count_serviced += count;

            if (ctx->idx > 0) {

                /* wait to be released into next iteration and updated count_serviced */
                while (thread_sub_sync[ctx->idx].l_coll_id < thread_sub_sync[ctx->idx].g_coll_id) {
                    /* busy wait */
                }
            }
            else {
                do {
                    inprogress = 0;
                    for (i = 1; i < ctx->nthreads; i++) {
                        if (thread_sub_sync[ctx->idx].g_coll_id > thread_sub_sync[i].g_coll_id) {
                            inprogress++;
                            break;
                        }
                    }
                } while(inprogress);

                for (i = 1; i < ctx->nthreads; i++) {
                    thread_sub_sync[i].l_coll_id++;
                }
                dpu_hc_reply(ctx->hc, ctx->ar_sync);
            }
        } while (ctx->ar_sync.count_serviced < lsync->count_total);

        thread_main_sync[ctx->idx].l_coll_id++;
    }

//     fprintf(stderr, "ctx->itt = %u\n", ctx->itt);
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
    thread_main_sync = aligned_alloc(64, nthreads * sizeof(*thread_main_sync));
    memset(thread_main_sync, 0, nthreads * sizeof(*thread_main_sync));

    thread_sub_sync = aligned_alloc(64, nthreads * sizeof(*thread_sub_sync));
    memset(thread_sub_sync, 0, nthreads * sizeof(*thread_sub_sync));

    memset(&tmp_sync, 0, sizeof(tmp_sync));

    dpu_hc_init(hc);
    dpu_hc_accept(hc);

    for(i = 0; i < nthreads; i++) {
//         printf("Thread %d spawned!\n", i);
        UCC_CHECK(dpu_ucc_alloc_team(&ucc_glob, &tctx_pool[i].comm));
        tctx_pool[i].idx = i;
        tctx_pool[i].nthreads = nthreads;
        tctx_pool[i].hc       = hc;
        tctx_pool[i].ar_sync.coll_id = 0;
        tctx_pool[i].ar_sync.count_serviced = 0;

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
