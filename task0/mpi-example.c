#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>

#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int np, self;
    struct utsname uts;
    const char *nodename;

    if (uname(&uts) == -1) {
        fprintf(stderr, "uname() failed: %s\n", strerror(errno));
        return 1;
    }

    nodename = uts.nodename;
    
    // In multi-threaded programs (e.g., when using OpenMP), one should
    // ask MPI to provide thread-support. We only require "serialized"
    // threading in MPI, i.e., only one thread at a time can make MPI
    // calls (we will use MPI only outside OpenMP or from blocks
    // annotated as "omp single").
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }

    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &self);

    printf("This is process #%d of %d in total on host %s.\n",
           self, np, nodename);

    // Open the OpenMP thread parallelism in each MPI process.
#pragma omp parallel
    {
	int np_omp   = omp_get_num_threads();
	int self_omp = omp_get_thread_num();
	printf("This is thread #%d of %d threads in process %d on host %s.\n",
               self_omp, np_omp, self, nodename);
    }

    MPI_Finalize();

    return 0;
}
