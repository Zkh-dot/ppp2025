#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "mpi.h"

#define TREE_SIZE 2

// Broadcast mode
enum { NONE, DIRECT, TREELIKE, NATIVE } mode;

static bool non_blocking; // whether to use non-blocking communication functions
static int sender;        // root process
static int size;          // number of elements to broadcast
static int times;         // number of repetitions
static int *buffer;       // buffer for data

static int np;   // number of MPI processes
static int self; // own process ID


// Returns the number of seconds since "the Epoch".
static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000.0;
}

static void usage(const char *progname) {
    fprintf(stderr,
            "USAGE: %s [-s EXPO] [-# TIMES] [-n | -d | -t] [-i]\n"
            "   -s EXPO   use buffer with 2^EXPO int values\n"
            "   -# TIMES  repeat broadcast TIMES times\n"
            "   -n        use native broadcast\n"
            "   -d        use direct broadcast\n"
            "   -t        use tree-like broadcast\n"
            "   -i        use MPI_Isend (with -d and -t)\n"
            "   -h        show this help\n"
            "\n",
            progname);
}

static void broadcast() {
    // please implement
    switch (mode) {
    case NONE:
        break;
    case NATIVE:
        MPI_Bcast(buffer, size, MPI_INT, sender, MPI_COMM_WORLD);
        break;
    case DIRECT:
        if(self == sender) {
            if(non_blocking) {
                MPI_Request *requests = malloc(np * sizeof(MPI_Request));
                int r = 0;
                for(int i = 0; i < np; i++) {
                    if(i != sender) {
                        MPI_Isend(buffer, size, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[r]);
                        r++;
                    }
                }
                MPI_Waitall(r, requests, MPI_STATUS_IGNORE);
                free(requests);
            } else {
                for(int i = 0; i < np; i++) {
                    if(i != sender) {
                        MPI_Send(buffer, size, MPI_INT, i, 0, MPI_COMM_WORLD);
                    }
                }
            }
        } 
        else {
            MPI_Recv(buffer, size, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        break;
    case TREELIKE:
        int parent = (self - 1) / TREE_SIZE;
        if(self != sender) {
            MPI_Recv(buffer, size, MPI_INT, parent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int child, r_size = TREE_SIZE;
        MPI_Request requests[TREE_SIZE];
        for(int i = 0; i < TREE_SIZE; i++) {
            child = TREE_SIZE * self + i + 1;
            if(child >= np) {
                r_size = i;
                break;
            }
            if(non_blocking) {
                MPI_Isend(buffer, size, MPI_INT, child, 0, MPI_COMM_WORLD, &requests[i]);
            } else {
                MPI_Send(buffer, size, MPI_INT, child, 0, MPI_COMM_WORLD);
            }
        }
        if(non_blocking){
            MPI_Waitall(r_size, requests, MPI_STATUS_IGNORE);
        }
        break;
    default:
        fprintf(stderr, "Unknown mode: %d\n", mode);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int c;
    int i;
    double start, end;
    bool ok;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    // Parse command line options
    size = 1024;
    sender = 0;
    times = 1;
    non_blocking = false;
    mode = NONE;
    while ((c = getopt(argc, argv, "dtnip:s:#:h")) != -1) {
        switch (c) {
        case 'd':
            mode = DIRECT;
            break;
        case 't':
            mode = TREELIKE;
            break;
        case 'n':
            mode = NATIVE;
            break;
        case 'i':
            non_blocking = 1;
            break;
        case 'p':
            sender = atoi(optarg);
            break;
        case 's':
            size = 1 << atoi(optarg);
            break;
        case '#':
            times = atoi(optarg);
            break;
        case 'h':
            usage(argv[0]);
            MPI_Finalize();
            return 0;
        default:
            if (self == 0)
                usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    buffer = malloc(sizeof(int) * size);
    if (buffer == NULL) {
        fprintf(stderr, "malloc failed\n");
        MPI_Finalize();
        return 1;
    }

    // Initialize data on "sender"
    if (self == sender) {
        printf("Buffer size: %zu KiB\n", (size_t)(size * sizeof(int)) / 1024);
        for (i = 0; i < size; i++)
            buffer[i] = i;
    } else {
        for (i = 0; i < size; i++)
            buffer[i] = -1;
    }

    // Perform "times" many broadcasts and measure execution time
    MPI_Barrier(MPI_COMM_WORLD);
    start = seconds();
    for (i = 0; i < times; i++)
        broadcast();
    MPI_Barrier(MPI_COMM_WORLD);
    end = seconds();

    // Check if every process has received the correct data
    ok = true;
    for (i = 0; i < size; i++)
        ok = ok && (buffer[i] == i);
    free(buffer);

    printf("Process %d: %s, time: %g secs\n", self, ok ? "ok" : "FAIL", (end - start) / times);

    MPI_Finalize();

    return 0;
}
