#include <getopt.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

#include "ppp/ppp.h"

static int MIN_COLOR_TEMP = 1000;
static int MAX_COLOR_TEMP = 10000;
static int DEFAULT_ORIGINAL_TEMP = 6600;
static int DEFAULT_NEW_TEMP = 1500;

static void usage(const char *progname) {
    int self;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    if (self == 0) {
        fprintf(stderr,
                "USAGE: %s -i image.pgm -o output.pgm "
                "[-k ORIG] [-n NEW] [-p] [-L] [-S]\n"
                "  ORIG  original color temperatue in Kelvin (default: %d)\n"
                "  NEW   new color temperature in Kelven (default %d)\n"
                "  -p    use parallel implementation (with MPI and OpenMP)\n"
                "  -L    load image in parallel (only with -p)\n"
                "  -S    save image in parallel (only with -p)\n"
                "  -h    show this help\n",
                progname, DEFAULT_ORIGINAL_TEMP, DEFAULT_NEW_TEMP);
    }
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }

    // Set efault parameter values.
    struct TaskInput TI;
    TI.filename = NULL;
    TI.outfilename = NULL;
    TI.originalColorTemp = DEFAULT_ORIGINAL_TEMP;
    TI.newColorTemp = DEFAULT_NEW_TEMP;
    TI.parallel_loading = false;
    TI.parallel_saving = false;
    bool parallel = false;

    int option;
    while ((option = getopt(argc, argv, "i:o:k:n:hLSp")) != -1) {
        switch (option) {
        case 'i':
            TI.filename = strdup(optarg);
            break;
        case 'o':
            TI.outfilename = strdup(optarg);
            break;
        case 'k':
            TI.originalColorTemp = atoi(optarg);
            break;
        case 'n':
            TI.newColorTemp = atoi(optarg);
            break;
        case 'L':
            TI.parallel_loading = true;
            break;
        case 'S':
            TI.parallel_saving = true;
            break;
        case 'p':
            parallel = true;
            break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (TI.filename == NULL) {
        fprintf(stderr, "Missing input file name\n");
        MPI_Finalize();
        return 1;
    }

    if (TI.outfilename == NULL) {
        fprintf(stderr, "Missing output file name\n");
        MPI_Finalize();
        return 1;
    }

    if (TI.originalColorTemp < MIN_COLOR_TEMP
        || TI.originalColorTemp > MAX_COLOR_TEMP
        || TI.newColorTemp < MIN_COLOR_TEMP
        || TI.newColorTemp > MAX_COLOR_TEMP) {
        fprintf(stderr, "Color temperatures must be between %d and %d.\n",
            MIN_COLOR_TEMP, MAX_COLOR_TEMP);
        MPI_Finalize();
        return 1;
    }

    if (parallel)
        compute_parallel(&TI);
    else
        compute_single(&TI);

    free(TI.outfilename);
    free(TI.filename);

    MPI_Finalize();

    return 0;
}
