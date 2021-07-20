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

#define DPU_PIPELINE_BUFFER_SIZE (4 * 1024 * 1024)
#define DPU_PIPELINE_BUFFERS     (2)

typedef struct {
    pthread_t       id;
    int             idx;
    int             nthreads;
    dpu_ucc_comm_t  comm;
    dpu_hc_t        *hc;
    unsigned long   pipeline_buffer_size;
    unsigned int    pipeline_buffers;
    unsigned int    buf_idx;
    dpu_get_sync_t  coll_sync;
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

static void dpu_thread_set_affinity(thread_ctx_t *ctx)
{
    int i;
    int places = CORES/ctx->nthreads;
    pthread_t thread = pthread_self();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
	for (i = 0; i < places; i++) {
		CPU_SET((ctx->idx*places)+i, &cpuset);
	}
    
    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

static void dpu_coll_init_allreduce(thread_ctx_t *ctx, ucc_coll_req_h *request)
{
    size_t dt_size = dpu_ucc_dt_size(tmp_sync.dtype);
    size_t count = DPU_MIN(ctx->pipeline_buffer_size/dt_size,
                tmp_sync.count_in - ctx->coll_sync.count_serviced);
    ctx->coll_sync.count_serviced += count;
    size_t block = count / ctx->nthreads;
    size_t offset = (ctx->buf_idx % ctx->pipeline_buffers) * ctx->pipeline_buffer_size + block * ctx->idx;

    
    if(ctx->idx < (count % ctx->nthreads)) {
        offset += ctx->idx;
        block++;
    } else {
        offset += (count % ctx->nthreads);
    }

    ucc_coll_args_t coll = {
        .coll_type = UCC_COLL_TYPE_ALLREDUCE,
        .mask      = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS,
        .src.info = {
            .buffer   = (char *)ctx->hc->mem_segs.put.base + offset,
            .count    = block,
            .datatype = tmp_sync.dtype,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .dst.info = {
            .buffer   = (char *)ctx->hc->mem_segs.get.base + offset,
            .count    = block,
            .datatype = tmp_sync.dtype,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .reduce = {
            .predefined_op = tmp_sync.op,
        },
    };

    UCC_CHECK(ucc_collective_init(&coll, request, ctx->comm.team));
}

static void dpu_coll_init_alltoall(thread_ctx_t *ctx, ucc_coll_req_h *request)
{
    /* Multithreading not supported */
    if(ctx->idx > 0) {
        *request = NULL;
        return;
    }
    
    ucc_coll_args_t coll = {
        .coll_type = UCC_COLL_TYPE_ALLTOALL,
        .src.info = {
            .buffer   = ctx->hc->mem_segs.put.base,
            .count    = tmp_sync.count_total,
            .datatype = tmp_sync.dtype,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .dst.info = {
            .buffer   = ctx->hc->mem_segs.get.base,
            .count    = tmp_sync.count_total,
            .datatype = tmp_sync.dtype,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
    };

    UCC_CHECK(ucc_collective_init(&coll, request, ctx->comm.team));
}

void *dpu_worker(void *arg)
{
    // fprintf (stdout, "sleeping %d\n", getpid());
    // sleep(20);

    thread_ctx_t *ctx = (thread_ctx_t*)arg;
    int i = 0, j = 0, inprogress = 0;
    dpu_put_sync_t *lsync = ctx->hc->mem_segs.sync.base;
    ucc_coll_req_h request;

    dpu_thread_set_affinity(ctx);

    while(1) {
        /* Wait for operation to start */
        ctx->coll_sync.coll_id++;
        ctx->coll_sync.count_serviced = 0;
        ctx->buf_idx = 0;

        if (ctx->idx > 0) {
            while (thread_main_sync[ctx->idx].g_coll_id < ctx->coll_sync.coll_id) {
                /* busy wait */
            }
        }
        else {
            /* Data arrived, main thread will synchronize sub threads */
            dpu_hc_wait(ctx->hc, ctx->coll_sync.coll_id);
            for (i = 0; i < ctx->nthreads; i++) {
                thread_main_sync[i].g_coll_id++;
            }
        }

        /* Hang up? */
        // fprintf(stderr, "coll id: %d, type: %d\n", lsync->coll_id, lsync->coll_type);
        if (lsync->coll_type == UCC_COLL_TYPE_LAST) {
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
                while(lsync->count_in <= ctx->coll_sync.count_serviced) {
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

            ucc_coll_type_t coll_type = tmp_sync.coll_type;
            if (coll_type == UCC_COLL_TYPE_ALLREDUCE) {
                dpu_coll_init_allreduce(ctx, &request);
            } else if (coll_type == UCC_COLL_TYPE_ALLTOALL) {
                dpu_coll_init_alltoall(ctx, &request);
            } else if (coll_type == UCC_COLL_TYPE_LAST) {
                fprintf(stderr, "Received hangup, exiting loop\n");
                break;
            } else {
                fprintf(stderr, "Unsupported coll type: %d\n", coll_type);
            }

            if (request != NULL) {
                UCC_CHECK(ucc_collective_post(request));
                while (UCC_OK != ucc_collective_test(request)) {
                    ucc_context_progress(ctx->comm.ctx);
                }
                UCC_CHECK(ucc_collective_finalize(request));
            }

            // unsigned long *p = (unsigned long *)ctx->hc->mem_segs.put.base;
            // unsigned long *p1 = (unsigned long*)ctx->hc->mem_segs.get.base;
            // printf ("put.base[0]=%lu, get.base[0]=%lu\n", *p, *p1);

            // unsigned long *p2 = (unsigned long *)ctx->hc->mem_segs.put.base + 128;
            // unsigned long *p3 = (unsigned long*)ctx->hc->mem_segs.get.base + 128;
            // printf ("put.base[128]=%lu, get.base[128]=%lu\n", *p2, *p3);

            ctx->buf_idx++;
            // ctx->coll_sync.count_serviced += tmp_sync.count_in - ctx->coll_sync.count_serviced;
            thread_sub_sync[ctx->idx].g_coll_id++;
            // fprintf(stderr, "count in: %lu, total: %lu, serviced: %lu\n",
            //             tmp_sync.count_in, tmp_sync.count_total, ctx->coll_sync.count_serviced);

            if (ctx->idx > 0) {
                /* wait to be released into next iteration */
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
                dpu_hc_reply(ctx->hc, ctx->coll_sync);
            }
        } while (ctx->coll_sync.count_serviced < lsync->count_total);

        thread_main_sync[ctx->idx].l_coll_id++;
        // fprintf(stderr, "l_coll_id: %d\n", thread_main_sync[ctx->idx].l_coll_id++);
    }
    return NULL;
}

int main(int argc, char **argv)
{
//     fprintf (stderr, "%s\n", __FUNCTION__);
//     sleep(20);
    unsigned long    pipeline_buffer_size = DPU_PIPELINE_BUFFER_SIZE;
    unsigned int     pipeline_buffers = DPU_PIPELINE_BUFFERS;
    int              nthreads = 0;
    int              i = 0;
    thread_ctx_t     *tctx_pool = NULL;
    dpu_hc_t         *hc = NULL;
    char             *env = NULL;

    dpu_ucc_global_t ucc_glob;
    dpu_hc_t         hc_b;

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

    env = getenv("DPU_PIPELINE_BUFFER_SIZE");
    if (NULL != env) {
        pipeline_buffer_size = atol(env);
    }

    env = getenv("DPU_PIPELINE_BUFFERS");
    if (NULL != env) {
        pipeline_buffers = atoi(env);
    }

    tctx_pool = calloc(nthreads, sizeof(*tctx_pool));
    UCC_CHECK(dpu_ucc_init(argc, argv, &ucc_glob));

    thread_main_sync = aligned_alloc(64, nthreads * sizeof(*thread_main_sync));
    memset(thread_main_sync, 0, nthreads * sizeof(*thread_main_sync));

    thread_sub_sync = aligned_alloc(64, nthreads * sizeof(*thread_sub_sync));
    memset(thread_sub_sync, 0, nthreads * sizeof(*thread_sub_sync));

    memset(&tmp_sync, 0, sizeof(tmp_sync));

    hc = &hc_b;

    dpu_hc_init(hc);
    dpu_hc_accept(hc);

    for(i = 0; i < nthreads; i++) {
        UCC_CHECK(dpu_ucc_alloc_team(&ucc_glob, &tctx_pool[i].comm));
        tctx_pool[i].idx = i;
        tctx_pool[i].nthreads = nthreads;
        tctx_pool[i].hc       = hc;
        tctx_pool[i].coll_sync.coll_id = 0;
        tctx_pool[i].coll_sync.count_serviced = 0;
        tctx_pool[i].pipeline_buffers = pipeline_buffers;
        tctx_pool[i].pipeline_buffer_size = pipeline_buffer_size;
        tctx_pool[i].buf_idx = 0;

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
