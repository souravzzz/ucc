/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_DPU_H_
#define UCC_TL_DPU_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include <ucp/api/ucp.h>
#include <limits.h>

#ifndef UCC_TL_DPU_DEFAULT_SCORE
#define UCC_TL_DPU_DEFAULT_SCORE 30
#endif

#define UCC_TL_DPU_TC_POLL      10
#define UCC_TL_DPU_TASK_REQS    10

#define UCC_TL_DPU_EXCHANGE_LENGTH_TAG 1ull
#define UCC_TL_DPU_EXCHANGE_RKEY_TAG 2ull
#define UCC_TL_DPU_EXCHANGE_ADDR_TAG 3ull

#define MAX_DPU_HOST_NAME 256

#define UCC_TL_DPU_PIPELINE_BLOCK_SIZE_MIN 1024
#define UCC_TL_DPU_PIPELINE_BUFFERS_MIN 2

typedef enum {
    UCC_TL_DPU_UCP_REQUEST_ACTIVE,
    UCC_TL_DPU_UCP_REQUEST_DONE,
} ucc_tl_dpu_request_status_t;

typedef struct ucc_tl_dpu_request {
  ucc_tl_dpu_request_status_t status;
} ucc_tl_dpu_request_t;

typedef struct ucc_tl_dpu_iface {
    ucc_tl_iface_t super;
} ucc_tl_dpu_iface_t;
extern ucc_tl_dpu_iface_t ucc_tl_dpu;

typedef struct ucc_tl_dpu_lib_config {
    ucc_tl_lib_config_t super;
    uint64_t            pipeline_buffer_size;
    uint32_t            pipeline_buffers;
} ucc_tl_dpu_lib_config_t;

typedef struct ucc_tl_dpu_context_config {
    ucc_tl_context_config_t super;
    uint32_t                server_port;
    char                    *server_hname;
    char                    *host_dpu_list;
} ucc_tl_dpu_context_config_t;

typedef struct ucc_tl_dpu_lib {
    ucc_tl_lib_t            super;
    ucc_tl_dpu_lib_config_t cfg;
} ucc_tl_dpu_lib_t;
UCC_CLASS_DECLARE(ucc_tl_dpu_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_dpu_context {
    ucc_tl_context_t            super;
    ucc_tl_dpu_context_config_t cfg;
    ucp_context_h               ucp_context;
    ucp_worker_h                ucp_worker;
    ucp_ep_h                    ucp_ep;
} ucc_tl_dpu_context_t;
UCC_CLASS_DECLARE(ucc_tl_dpu_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_dpu_put_sync_t {
    ucc_datatype_t           dtype;
    ucc_reduction_op_t       op;
    ucc_coll_type_t          coll_type;
    unsigned int             count_total;
    unsigned int             count_out;
    unsigned int             coll_id;
} ucc_tl_dpu_put_sync_t;

typedef struct ucc_tl_dpu_get_sync_t {
    uint32_t    count_serviced;
    uint32_t    coll_id;
} ucc_tl_dpu_get_sync_t;

typedef struct ucc_tl_dpu_put_request {
    ucc_tl_dpu_request_t *data_req;
    ucc_tl_dpu_request_t *sync_req;
    ucc_tl_dpu_put_sync_t sync_data;
} ucc_tl_dpu_put_request_t;

typedef struct ucc_tl_dpu_get_request {
    ucc_tl_dpu_request_t *data_req;
} ucc_tl_dpu_get_request_t;

typedef struct ucc_tl_dpu_connect_s {
    ucp_mem_map_params_t    mmap_params;
    void                    *get_sync_rkey_buf;
    size_t                  get_sync_rkey_buf_size;
    size_t                  rem_rkeys_lengths[3];
    void                    *rem_rkeys;
    uint64_t                rem_addresses[3];
} ucc_tl_dpu_conn_buf_t;

typedef struct ucc_tl_dpu_team {
    ucc_tl_team_t         super;
    ucc_status_t          status;
    ucc_rank_t            size;
    ucc_rank_t            rank;
    uint32_t              coll_id;
    ucc_tl_dpu_get_sync_t get_sync;
    ucp_mem_h             get_sync_memh;
    uint64_t              rem_ctrl_seg;
    ucp_rkey_h            rem_ctrl_seg_key;
    uint64_t              *rem_data_in;
    ucp_rkey_h            rem_data_in_key;
    uint64_t              *rem_data_out;
    ucp_rkey_h            rem_data_out_key;
    ucc_tl_dpu_request_t  *send_req[3];
    ucc_tl_dpu_request_t  *recv_req[2];
    ucc_tl_dpu_conn_buf_t *conn_buf;
} ucc_tl_dpu_team_t;
UCC_CLASS_DECLARE(ucc_tl_dpu_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

typedef struct ucc_tl_dpu_task_req_t {
    ucp_request_param_t      req_param;
    ucc_tl_dpu_put_request_t *put_reqs;
    ucc_tl_dpu_get_request_t *get_reqs;
    uint32_t                 put_data_count;
    uint32_t                 get_data_count;
    uint32_t                 put_bf_idx;
    uint32_t                 get_bf_idx;
    uint32_t                 puts_in_flight;
} ucc_tl_dpu_task_req_t;

typedef struct ucc_tl_dpu_task {
    ucc_coll_task_t          super;
    ucc_coll_args_t          args;
    ucc_tl_dpu_team_t        *team;
    ucc_tl_dpu_put_sync_t    put_sync;
    ucc_tl_dpu_get_sync_t    get_sync;
    ucc_tl_dpu_task_req_t    task_reqs;
    size_t                   block_count;
    size_t                   block_data_size;
    uint32_t                 pipeline_buffers;
} ucc_tl_dpu_task_t;

typedef struct ucc_tl_dpu_config {
    ucc_tl_lib_config_t super;
} ucc_tl_dpu_config_t;

typedef struct ucc_tl_dpu {
    ucc_tl_lib_t        super;
    ucc_tl_dpu_config_t config;
} ucc_tl_dpu_t;

#define UCC_TL_DPU_SUPPORTED_COLLS \
    (UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_ALLTOALL)

#define UCC_TL_DPU_TEAM_LIB(_team)                                          \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_dpu_lib_t))

#define UCC_TL_DPU_TEAM_CTX(_team)                                          \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_dpu_context_t))

#define UCC_TL_DPU_TEAM_CORE_CTX(_team)                                     \
    ((_team)->super.super.context->ucc_context)

void ucc_tl_dpu_req_init(void *request);
void ucc_tl_dpu_req_cleanup(void * request);

ucc_status_t ucc_tl_dpu_req_test(ucc_tl_dpu_request_t **req, ucp_worker_h worker);
ucc_status_t ucc_tl_dpu_req_check(ucc_tl_dpu_team_t *team,
                                      ucc_tl_dpu_request_t *req);

void ucc_tl_dpu_send_handler_nbx(void *request, ucs_status_t status, void *user_data);
void ucc_tl_dpu_recv_handler_nbx(void *request, ucs_status_t status,
                      const ucp_tag_recv_info_t *tag_info,
                      void *user_data);

#endif