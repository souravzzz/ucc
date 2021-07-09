/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"
#include "tl_dpu_coll.h"

#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

void ucc_tl_dpu_send_handler_nbx(void *request, ucs_status_t status,
                                 void *user_data)
{
    ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
    req->status = UCC_TL_DPU_UCP_REQUEST_DONE;
}

void ucc_tl_dpu_recv_handler_nbx(void *request, ucs_status_t status,
                      const ucp_tag_recv_info_t *tag_info,
                      void *user_data)
{
  ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
  req->status = UCC_TL_DPU_UCP_REQUEST_DONE;
}

static ucc_tl_dpu_task_t * ucc_tl_dpu_alloc_task(void)
{
    ucc_tl_dpu_task_t *task =
        (ucc_tl_dpu_task_t *) ucc_calloc(1, sizeof(ucc_tl_dpu_task_t),
                                         "Allocate task");
    return task;
}

static ucc_status_t ucc_tl_dpu_free_task(ucc_tl_dpu_task_t *task)
{
    ucc_tl_dpu_put_request_t *put_req = task->task_reqs.put_reqs, *put_req_tmp;
    ucc_tl_dpu_get_request_t *get_req = task->task_reqs.get_reqs, *get_req_tmp;
    
    while (NULL != put_req) {
        put_req_tmp = put_req;
        put_req = put_req->next;
        ucc_free(put_req_tmp);
    }
    while (NULL != get_req) {
        get_req_tmp = get_req;
        get_req = get_req->next;
        ucc_free(get_req_tmp);
    }

    ucc_free(task);
    return UCC_OK;
}

void ucc_tl_dpu_req_init(void* request)
{
    ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
    req->status = UCC_TL_DPU_UCP_REQUEST_ACTIVE;
}

void ucc_tl_dpu_req_cleanup(void* request){ 
    return;
}

ucc_status_t ucc_tl_dpu_req_test(ucc_tl_dpu_request_t **req,
                                 ucp_worker_h worker) {
    if (*req == NULL) {
        return UCC_OK;
    }

    if ((*req)->status == UCC_TL_DPU_UCP_REQUEST_DONE) {
        (*req)->status = UCC_TL_DPU_UCP_REQUEST_ACTIVE;
        ucp_request_free(*req);
        (*req) = NULL;
        return UCC_OK;
    }
    ucp_worker_progress(worker);
    return UCC_INPROGRESS;
}

inline
ucc_status_t ucc_tl_dpu_req_check(ucc_tl_dpu_team_t *team,
                                      ucc_tl_dpu_request_t *req) {
    if (UCS_PTR_IS_ERR(req)) {
        tl_error(team->super.super.context->lib,
                 "failed to send/recv msg");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_issue_put( ucc_tl_dpu_task_t *task,
    ucc_tl_dpu_context_t *ctx, ucc_tl_dpu_team_t *team, void *sbuf,
    size_t offset, uint32_t bf_idx, size_t count, size_t data_size,
    ucp_request_param_t *req_param)
{
    unsigned int buff_idx = task->task_reqs.put_bf_idx;
    ucc_tl_dpu_put_request_t *put_req = &task->task_reqs.put_reqs[buff_idx];

    ucp_worker_fence(ctx->ucp_worker);
    put_req->data_req =
        ucp_put_nbx(ctx->ucp_ep, ((char *)sbuf + offset), data_size,
                    team->rem_data_in[bf_idx], team->rem_data_in_key,
                    req_param);
    if (ucc_tl_dpu_req_check(team, put_req->data_req) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }
    
    ucp_worker_fence(ctx->ucp_worker);
    
    /* Initialize put_sync_data for xfer */
    task->put_sync.count_out += count;
    memcpy(&put_req->sync_data, &task->put_sync, sizeof(task->put_sync));

    put_req->sync_req =
        ucp_put_nbx(ctx->ucp_ep, &put_req->sync_data , sizeof(task->put_sync),
                    team->rem_ctrl_seg, team->rem_ctrl_seg_key,
                    req_param);
    if (ucc_tl_dpu_req_check(team, put_req->sync_req) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_issue_get(
    ucc_tl_dpu_task_t *task, ucc_tl_dpu_context_t *ctx,
    ucc_tl_dpu_team_t *team, void *rbuf, size_t offset,
    uint32_t bf_idx, size_t data_size, ucp_request_param_t *req_param)
{
    unsigned int buff_idx = task->task_reqs.get_bf_idx;
    ucc_tl_dpu_get_request_t *get_req = &task->task_reqs.get_reqs[buff_idx];

    get_req->data_req =
        ucp_get_nbx(ctx->ucp_ep, ((char *)rbuf + offset), data_size,
                    team->rem_data_out[bf_idx], team->rem_data_out_key,
                    req_param);
    if (ucc_tl_dpu_req_check(team, get_req->data_req) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }
    ucp_worker_fence(ctx->ucp_worker);

    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_check_progress(
    ucc_tl_dpu_task_t *task, ucc_tl_dpu_context_t *ctx)
{
    int i = 0, coll_poll = UCC_TL_DPU_COLL_POLL;
    ucc_tl_dpu_put_request_t *put_req;
    ucc_tl_dpu_get_request_t *get_req;
    ucc_status_t status;

    ucp_worker_progress(ctx->ucp_worker);

    for (i = 0; i < coll_poll; i++) {
        status = UCC_OK;

        put_req = task->task_reqs.put_reqs;
        while (NULL != put_req && UCC_OK == status) {
            status |= ucc_tl_dpu_req_test(&put_req->data_req, ctx->ucp_worker);
            status |= ucc_tl_dpu_req_test(&put_req->sync_req, ctx->ucp_worker);
            put_req = put_req->next;
        }

        get_req = task->task_reqs.get_reqs;
        while (NULL != get_req && UCC_OK == status) {
            status |= ucc_tl_dpu_req_test(&get_req->data_req,
                                              ctx->ucp_worker);
            get_req = get_req->next;
        }

        if (UCC_OK == status) {
            break;
        }
    }

    /* Ops are done, or pipeline is progressing, check count */
    if (UCC_OK == status) {
        status =
            (task->task_reqs.put_data_count + task->task_reqs.get_data_count) ==
             (2 * task->args.src.info.count) ? UCC_OK : UCC_INPROGRESS;
    }

    return status;
}

ucc_status_t ucc_tl_dpu_allreduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t       *task =
        ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_team_t       *team       = task->team;
    ucc_tl_dpu_context_t    *ctx        = UCC_TL_DPU_TEAM_CTX(team);
    void                    *sbuf       = task->args.src.info.buffer;
    void                    *rbuf       = task->args.dst.info.buffer;
    size_t                  count_total = task->args.src.info.count;
    size_t                  count_rem   = count_total -
                                          task->task_reqs.put_data_count;
    ucc_datatype_t          dt          = task->args.src.info.datatype;
    size_t                  dt_size     = ucc_dt_size(dt);
    size_t                  block_count = task->block_count;
    ucp_request_param_t     *req_param  = &task->task_reqs.req_param;
    size_t                  block_data_size;
    ucc_status_t            status;

    if (task->task_reqs.put_bf_idx == task->task_reqs.get_bf_idx &&
        task->task_reqs.put_data_count < count_total) {

        block_count = ucc_min(block_count, count_rem);
        block_data_size = block_count * dt_size;

        status = ucc_tl_dpu_issue_put(task, ctx, team, sbuf,
                    task->task_reqs.put_data_count * dt_size /*offset */,
                    task->task_reqs.put_bf_idx /* buffer idx */,
                    block_count, block_data_size, req_param);
        if (UCC_OK != status) {
            goto err;
        }
        task->task_reqs.put_data_count += block_count;
        task->task_reqs.put_bf_idx = (task->task_reqs.put_bf_idx + 1) %
            task->pipeline_buffers;
    }

    if (team->coll_id == (team->get_sync.coll_id + 1) &&
        task->task_reqs.get_data_count < team->get_sync.count_serviced) {
        block_count = ucc_min(block_count, team->get_sync.count_serviced -
                                           task->task_reqs.get_data_count);
        block_data_size = block_count * dt_size;
        status = ucc_tl_dpu_issue_get(task, ctx, team, rbuf,
                    task->task_reqs.get_data_count * dt_size /*offset*/,
                    task->task_reqs.get_bf_idx /* buffer idx */,
                    block_data_size, req_param);
        if (UCC_OK != status) {
            goto err;
        }
        task->task_reqs.get_data_count += block_count;
        task->task_reqs.get_bf_idx = (task->task_reqs.get_bf_idx + 1) %
            task->pipeline_buffers;
    }

    status = ucc_tl_dpu_check_progress(task, ctx);
    task->super.super.status = status;

err:
    return status;
}

ucc_status_t ucc_tl_dpu_allreduce_start(ucc_coll_task_t *coll_task)
{
    // fprintf (stdout, "sleeping %d\n", getpid());
    // // sleep(20);

    ucc_tl_dpu_task_t    *task        = ucs_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_team_t    *team        = task->team;
    ucc_tl_dpu_context_t *ctx         = UCC_TL_DPU_TEAM_CTX(team);
    void                 *sbuf        = task->args.src.info.buffer;
    void                 *rbuf        = task->args.dst.info.buffer;
    size_t               count_total  = task->args.src.info.count;
    ucc_datatype_t       dt           = task->args.src.info.datatype;
    size_t               dt_size      = ucc_dt_size(dt);
    size_t               block_count, block_data_size;
    ucp_request_param_t  *req_param;
    ucc_status_t         status;
 
    tl_info(team->super.super.context->lib, "Collective post");

    if (UCC_IS_INPLACE(task->args)) {
        sbuf = task->args.src.info.buffer = rbuf;
    }
    
    block_count = task->block_count = ucc_min(
        (UCC_TL_DPU_TEAM_LIB(team)->cfg.pipeline_buffer_size) / dt_size,
        count_total);
    block_data_size = task->block_data_size = task->block_count * dt_size;

    req_param = &task->task_reqs.req_param;
    req_param->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_DATATYPE;
    req_param->datatype     = ucp_dt_make_contig(1);
    req_param->cb.send      = ucc_tl_dpu_send_handler_nbx;
    req_param->cb.recv      = ucc_tl_dpu_recv_handler_nbx;

    /* XXX set memory
    req_param.mask          = 0;
    req_param.mem_type      = task->args.src.info.mem_type;
    req_param.memory_type   = ucc_memtype_to_ucs[mtype];
    */

   /* First put */
    status = ucc_tl_dpu_issue_put(task, ctx, team, sbuf,
                0 /*offset*/, 0 /* bf idx */,
                block_count, block_data_size, req_param);
    if (UCC_OK != status) {
        goto put_err;
    }
    task->task_reqs.put_data_count += block_count;
    task->task_reqs.put_bf_idx =
        (task->task_reqs.put_bf_idx + 1) % task->pipeline_buffers;

    status = ucc_tl_dpu_check_progress(task, ctx);
    task->super.super.status = status;

    if (UCC_INPROGRESS == status) {
        status = ucc_tl_dpu_allreduce_progress(&task->super);
        if (UCC_INPROGRESS == status) {
            ucc_progress_enqueue(UCC_TL_DPU_TEAM_CORE_CTX(team)->pq, &task->super);
            return UCC_OK;
        }
    }

    return UCC_OK;
put_err:
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_dpu_allreduce_init(ucc_tl_dpu_task_t *task)
{
    ucc_coll_args_t     *coll_args = &task->args;
    ucc_tl_dpu_team_t   *team      = task->team;

    if (task->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "userdefined reductions are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (!UCC_IS_INPLACE(task->args) && (task->args.src.info.mem_type !=
                                        task->args.dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yetpp");
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Set sync information for DPU */
    task->put_sync.coll_id           = team->coll_id;
    task->put_sync.dtype             = coll_args->src.info.datatype;
    task->put_sync.count_total       = coll_args->src.info.count;
    task->put_sync.count_out         = 0;
    task->put_sync.op                = coll_args->reduce.predefined_op;
    task->put_sync.coll_type         = coll_args->coll_type;
    task->get_sync.coll_id           = team->coll_id;
    task->get_sync.count_serviced    = 0;
    task->pipeline_buffers           =
        (UCC_TL_DPU_TEAM_LIB(task->team)->cfg.pipeline_buffers);

    /* Initialize all request stuff for pipelining */
    memset(&task->task_reqs.req_param, 0, sizeof(ucp_request_param_t));
    task->task_reqs.put_reqs         = ucc_calloc(task->pipeline_buffers, sizeof(ucc_tl_dpu_put_request_t));
    task->task_reqs.get_reqs         = ucc_calloc(task->pipeline_buffers, sizeof(ucc_tl_dpu_get_request_t));
    task->task_reqs.put_data_count   = 0;
    task->task_reqs.get_data_count   = 0;
    task->task_reqs.put_bf_idx       = 0;
    task->task_reqs.get_bf_idx       = 0;

    task->super.post     = ucc_tl_dpu_allreduce_start;
    task->super.progress = ucc_tl_dpu_allreduce_progress;

    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t *task = ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    tl_info(task->team->super.super.context->lib, "finalizing task %p", task);
    ucc_tl_dpu_free_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_coll_init(ucc_base_coll_args_t      *coll_args,
                                         ucc_base_team_t    *team,
                                         ucc_coll_task_t    **task_h)
{
    ucc_tl_dpu_team_t    *tl_team = ucc_derived_of(team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_task_t    *task    = ucc_tl_dpu_alloc_task();
    ucc_status_t          status  = UCC_OK;

    ucc_coll_task_init(&task->super);
    tl_info(team->context->lib, "task %p initialized", task);

    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));

    /* Misc init stuff */
    task->team                       = tl_team;
    task->super.finalize             = ucc_tl_dpu_coll_finalize;
    task->super.triggered_post       = NULL;

    /* Increase your inflight collective count */
    tl_team->coll_id++;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_dpu_allreduce_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        ucc_tl_dpu_free_task(task);
        return status;
    }

    tl_info(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}
