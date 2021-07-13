#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

MPI_DIR=/global/scratch/users/sourav/build-arm/ompi
UCX_DIR=/global/scratch/users/sourav/build-arm/ucx
UCC_DIR=/global/scratch/users/sourav/build-arm/ucc
MPICC=$(MPI_DIR)/bin/mpicc

CFLAGS = -I$(UCC_DIR)/include -I$(UCX_DIR)/include
LDFLAGS = -L$(UCC_DIR)/lib $(UCC_DIR)/lib/libucc.so $(UCX_DIR)/lib/libucs.so $(UCX_DIR)/lib/libucp.so -Wl,-rpath -Wl,$(UCC_DIR)/lib -Wl,-rpath -Wl,$(UCC_DIR)/lib

rel:
	$(MPICC) -O3 -DNDEBUG -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_ucc.c $(LDFLAGS)

dbg:
	$(MPICC) -O0 -g -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_ucc.c $(LDFLAGS)

clean:
	rm -f dpu_server
