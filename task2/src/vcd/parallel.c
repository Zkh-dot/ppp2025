#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

inline static double grayvalueToDouble(uint8_t v, int maxcolor) {
    return (double)v / maxcolor;
}

inline static int grayvalueFromDouble(double d, int maxcolor) {
    int v = lrint(d * maxcolor);
    return (v < 0 ? 0 : (v > maxcolor ? maxcolor : v));
}

static void swap(double **a, double **b) {
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

static void broadcast_image_info(int *rows, int *columns, int *maxcolor, enum pnm_kind *kind) {
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kind, sizeof(enum pnm_kind), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void calculate_row_distribution(int total_rows, int np, int rank,
                                       int *local_rows, int *start_row) {
    int base_rows = total_rows / np;
    int remainder = total_rows % np;

    if (rank < remainder) {
        *local_rows = base_rows + 1;
        *start_row = rank * (base_rows + 1);
    } else {
        *local_rows = base_rows;
        *start_row = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }
}

static void distribute_image_data(double *full_image, double *local_image,
                                  int rows, int columns, int np, int rank) {
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(np * sizeof(int));
        displs = malloc(np * sizeof(int));

        for (int i = 0; i < np; i++) {
            int proc_rows, start_row;
            calculate_row_distribution(rows, np, i, &proc_rows, &start_row);
            sendcounts[i] = proc_rows * columns;
            displs[i]    = start_row * columns;
        }
    }

    int local_rows, start_row;
    calculate_row_distribution(rows, np, rank, &local_rows, &start_row);
    int local_size = local_rows * columns;

    MPI_Scatterv(full_image, sendcounts, displs, MPI_DOUBLE,
                 local_image, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

static void collect_image_data(double *local_image, double *full_image,
                               int rows, int columns, int np, int rank) {
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = malloc(np * sizeof(int));
        displs = malloc(np * sizeof(int));

        for (int i = 0; i < np; i++) {
            int proc_rows, start_row;
            calculate_row_distribution(rows, np, i, &proc_rows, &start_row);
            recvcounts[i] = proc_rows * columns;
            displs[i]     = start_row * columns;
        }
    }

    int local_rows, start_row;
    calculate_row_distribution(rows, np, rank, &local_rows, &start_row);
    int local_size = local_rows * columns;

    MPI_Gatherv(local_image, local_size, MPI_DOUBLE,
                full_image, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

static uint8_t* load_and_validate_image(const char* filename, enum pnm_kind *kind,
                                        int *rows, int *columns, int *maxcolor) {
    uint8_t *image = ppp_pnm_read(filename, kind, rows, columns, maxcolor);

    if (image == NULL) {
        fprintf(stderr, "Could not load image from file '%s'.\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else if (*kind != PNM_KIND_PGM) {
        fprintf(stderr, "Image is not a \"portable graymap.\" (PGM)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return image;
}

static void exchange_ghost_rows(double *local_data, int local_rows, int columns,
                                int rank, int np) {
    MPI_Request requests[4];
    int req_count = 0;

    int ghost_top = (rank > 0) ? 1 : 0;
    int ghost_bottom = (rank < np - 1) ? 1 : 0;

    
    if (rank > 0) {
        
        MPI_Isend(&local_data[ghost_top * columns], columns, MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        
        MPI_Irecv(&local_data[0], columns, MPI_DOUBLE,
                  rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    
    if (rank < np - 1) {
        
        MPI_Isend(&local_data[(local_rows - 1 + ghost_top) * columns], columns, MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        MPI_Irecv(&local_data[(local_rows + ghost_top) * columns], columns, MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

inline static double get_pixel_safe(double *local_data,
                                    int x, int y,
                                    int local_rows, int columns,
                                    int ghost_top, int ghost_bottom) {
    int yy = y + ghost_top;
    int total_rows = local_rows + ghost_top + ghost_bottom;
    if (x < 0 || x >= columns || yy < 0 || yy >= total_rows) {
        return 0.0;
    }
    return local_data[yy * columns + x];
}

#define phi(nu)                                      \
    ({                                              \
        const double chi = (nu) / kappa;            \
        chi * exp(-chi * chi / 2.0);                \
    })
#define xi(nu)                                      \
    ({                                              \
        const double psi = (nu) / (kappa * sqrt(2));\
        (1.0 / sqrt(2.0)) * psi * exp(-psi * psi / 2.0);\
    })

static double compute_vcd_delta_for_pixel(double *image,
                                          int x, int y,
                                          int local_rows, int columns,
                                          double kappa,
                                          int ghost_top, int ghost_bottom) {
    double current = get_pixel_safe(image, x, y, local_rows, columns, ghost_top, ghost_bottom);

    
    double diff_right = get_pixel_safe(image, x + 1, y, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_left  = current - get_pixel_safe(image, x - 1, y, local_rows, columns, ghost_top, ghost_bottom);
    double diff_down  = get_pixel_safe(image, x, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_up    = current - get_pixel_safe(image, x, y - 1, local_rows, columns, ghost_top, ghost_bottom);

    
    double diff_diag1 = get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_diag2 = current - get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom);
    double diff_diag3 = get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_diag4 = current - get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom);

    
    return phi(diff_right) - phi(diff_left)
         + phi(diff_down)  - phi(diff_up)
         + xi(diff_diag1)  - xi(diff_diag2)
         + xi(diff_diag3)  - xi(diff_diag4);
}

static double perform_vcd_iteration_distributed(double **image,
                                               double **temp,
                                               int local_rows,
                                               int columns,
                                               double kappa,
                                               double dt,
                                               int rank,
                                               int np,
                                               int global_start_row,
                                               int total_rows) {
    
    exchange_ghost_rows(*image, local_rows, columns, rank, np);

    double local_deltaMax = 0.0;
    int ghost_top = (rank > 0) ? 1 : 0;
    int ghost_bottom = (rank < np - 1) ? 1 : 0;

    #pragma omp parallel for collapse(2) reduction(max: local_deltaMax) schedule(static)
    for (int y = 0; y < local_rows; y++) {
        for (int x = 0; x < columns; x++) {
            double delta = compute_vcd_delta_for_pixel(*image, x, y,
                                                       local_rows, columns,
                                                       kappa, ghost_top, ghost_bottom);
            int idx = (y + ghost_top) * columns + x;
            (*temp)[idx] = (*image)[idx] + kappa * dt * delta;

            
            
            int global_y = global_start_row + y;
            if (global_y > 0 && global_y < total_rows - 1 && x > 0 && x < columns - 1) {
                double abs_delta = fabs(delta);
                if (abs_delta > local_deltaMax) {
                    local_deltaMax = abs_delta;
                }
            }
        }
    }

    
    swap(image, temp);
    return local_deltaMax;
}

static double compute_ghostfree_boundary_columns(double *image,
                                                 double *temp,
                                                 int local_rows,
                                                 int columns,
                                                 double kappa,
                                                 double dt,
                                                 int ghost_top,
                                                 int ghost_bottom,
                                                 int global_start_row,
                                                 int total_rows) {
    double local_deltaMax = 0.0;

    int start_row = ghost_top;
    int end_row = local_rows - ghost_bottom;

    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < columns; x += columns - 1) { 
            double delta = compute_vcd_delta_for_pixel(image, x, y,
                                                       local_rows, columns,
                                                       kappa,
                                                       ghost_top, ghost_bottom);
            temp[y * columns + x] = image[y * columns + x] + delta * dt;
            local_deltaMax = fmax(local_deltaMax, fabs(delta));
        }
    }

    return local_deltaMax;
}

typedef struct {
    MPI_Request reqs[4];
} GhostExchange;

static void exchange_ghost_rows_nonblocking(
    double *image, int local_rows, int columns,
    int rank, int np,
    double *ghost_top, double *ghost_bottom,
    GhostExchange *ex)
{
    int above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int below = (rank < np - 1) ? rank + 1 : MPI_PROC_NULL;
    int count = columns;
    
    MPI_Irecv(ghost_top,    count, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &ex->reqs[0]);
    MPI_Irecv(ghost_bottom, count, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &ex->reqs[1]);
    
    MPI_Isend(image + columns,            count, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &ex->reqs[2]);
    MPI_Isend(image + (local_rows)*columns, count, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &ex->reqs[3]);
}

static double compute_ghost_dependent_rows(double *image,
                                           double *temp,
                                           int local_rows,
                                           int columns,
                                           double kappa,
                                           double dt,
                                           int ghost_top,
                                           int ghost_bottom,
                                           int global_start_row,
                                           int total_rows) {
    double local_deltaMax = 0.0;

    if (ghost_top) {
        int y = 0;
        for (int x = 0; x < columns; ++x) {
            double delta = compute_vcd_delta_for_pixel(image, x, y,
                                                       local_rows, columns,
                                                       kappa,
                                                       ghost_top, ghost_bottom);
            temp[y * columns + x] = image[y * columns + x] + delta * dt;
            local_deltaMax = fmax(local_deltaMax, fabs(delta));
        }
    }

    if (ghost_bottom) {
        int y = local_rows - 1;
        for (int x = 0; x < columns; ++x) {
            double delta = compute_vcd_delta_for_pixel(image, x, y,
                                                       local_rows, columns,
                                                       kappa,
                                                       ghost_top, ghost_bottom);
            temp[y * columns + x] = image[y * columns + x] + delta * dt;
            local_deltaMax = fmax(local_deltaMax, fabs(delta));
        }
    }

    return local_deltaMax;
}
static double perform_vcd_iteration_optimized(double **image,
                                              double **temp,
                                              int local_rows,
                                              int columns,
                                              double kappa,
                                              double dt,
                                              int rank,
                                              int np,
                                              int global_start_row,
                                              int total_rows,
                                              MPI_Request *ghost_requests,
                                              int *ghost_req_count) {
    int ghost_top = (rank > 0) ? 1 : 0;
    int ghost_bottom = (rank < np - 1) ? 1 : 0;

    
    exchange_ghost_rows(*image, local_rows, columns, rank, np);

    
    double local_deltaMax = 0.0;
    int start_row = ghost_top + 1;
    int end_row = local_rows - ghost_bottom - 1;

    for (int y = start_row; y <= end_row; ++y) {
        for (int x = 1; x < columns - 1; ++x) {
            double delta = compute_vcd_delta_for_pixel(*image, x, y,
                                                       local_rows, columns,
                                                       kappa,
                                                       ghost_top, ghost_bottom);
            (*temp)[y * columns + x] = (*image)[y * columns + x] + delta * dt;
            local_deltaMax = fmax(local_deltaMax, fabs(delta));
        }
    }

    
    double side_column_deltaMax = compute_ghostfree_boundary_columns(*image, *temp,
                                                                     local_rows, columns,
                                                                     kappa, dt,
                                                                     ghost_top, ghost_bottom,
                                                                     global_start_row, total_rows);
    local_deltaMax = fmax(local_deltaMax, side_column_deltaMax);

    
    if (*ghost_req_count > 0) {
        MPI_Waitall(*ghost_req_count, ghost_requests, MPI_STATUSES_IGNORE);
    }

    
    double ghost_row_deltaMax = compute_ghost_dependent_rows(*image, *temp,
                                                             local_rows, columns,
                                                             kappa, dt,
                                                             ghost_top, ghost_bottom,
                                                             global_start_row, total_rows);
    local_deltaMax = fmax(local_deltaMax, ghost_row_deltaMax);

    
    swap(image, temp);

    return local_deltaMax;
}

void vcd_parallel_v2(
    double **image,
    double **temp,
    int local_rows,
    int columns,
    const struct TaskInput *TI,
    int rank,
    int np,
    int global_start_row,
    int total_rows)
{
    
    double *ghost_top    = malloc(columns * sizeof(double));
    double *ghost_bottom = malloc(columns * sizeof(double));
    GhostExchange ex;
    int ghost_req_count = 4;

    for (int iter = 0; iter < TI->vcdN; ++iter) {
        
        exchange_ghost_rows_nonblocking(
            *image, local_rows, columns,
            rank, np,
            ghost_top, ghost_bottom,
            &ex
        );

        
        double delta_loc = perform_vcd_iteration_optimized(
            image, temp,
            local_rows, columns,
            TI->vcdKappa, TI->vcdDt,
            rank, np,
            global_start_row, total_rows,
            ex.reqs, &ghost_req_count
        );

        
        swap(image, temp);

        
        double delta_glb;
        MPI_Allreduce(&delta_loc, &delta_glb, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (delta_glb < TI->vcdEpsilon) break;
    }

    free(ghost_top);
    free(ghost_bottom);
}

void vcd_parallel(double **image,
                  double **temp,
                  int local_rows,
                  int columns,
                  const struct TaskInput *TI,
                  int rank,
                  int np,
                  int global_start_row,
                  int total_rows) {
    const double kappa = TI->vcdKappa;
    const double epsilon = TI->vcdEpsilon;
    const double dt = TI->vcdDt;
    const int N = TI->vcdN;

    int iteration = 0;
    double global_deltaMax = epsilon + 1.0;

    while (iteration < N && global_deltaMax > epsilon) {
        iteration++;
        double local_deltaMax = perform_vcd_iteration_distributed(image, temp,
                                                                 local_rows, columns,
                                                                 kappa, dt,
                                                                 rank, np,
                                                                 global_start_row,
                                                                 total_rows);

        
        MPI_Allreduce(&local_deltaMax, &global_deltaMax,
                      1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (TI->debugOutput && rank == 0) {
            printf("VCD Iteration %2d: max |Δ| = %g\n", iteration, global_deltaMax);
        }
    }
}

static void compute_sobel_gradients_distributed(double *image,
                                                int x, int y,
                                                int local_rows,
                                                int columns,
                                                double *sx,
                                                double *sy,
                                                int ghost_top,
                                                int ghost_bottom) {
    
    *sx =  get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         + 2 * get_pixel_safe(image, x    , y - 1, local_rows, columns, ghost_top, ghost_bottom)
         +   get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom)
         - 2 * get_pixel_safe(image, x    , y + 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom);

    
    *sy =  get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         + 2 * get_pixel_safe(image, x - 1, y    , local_rows, columns, ghost_top, ghost_bottom)
         +   get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         - 2 * get_pixel_safe(image, x + 1, y    , local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom);
}

static void apply_sobel_operator_distributed(double **input,
                                             double **temp,
                                             int local_rows,
                                             int columns,
                                             double sobelC,
                                             int rank,
                                             int np) {
    
    exchange_ghost_rows(*input, local_rows, columns, rank, np);

    int ghost_top = (rank > 0) ? 1 : 0;
    int ghost_bottom = (rank < np - 1) ? 1 : 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < local_rows; ++y) {
        for (int x = 0; x < columns; ++x) {
            double sx_val, sy_val;
            compute_sobel_gradients_distributed(*input, x, y,
                                               local_rows, columns,
                                               &sx_val, &sy_val,
                                               ghost_top, ghost_bottom);
            int idx = (y + ghost_top) * columns + x;
            (*temp)[idx] = sobelC * hypot(sx_val, sy_val);
        }
    }

    
    swap(input, temp);
}

void sobel_parallel(double **input, double **temp,
                    int local_rows, int columns,
                    double sobelC, int rank, int np) {
    apply_sobel_operator_distributed(input, temp, local_rows, columns, sobelC, rank, np);
}

static void print_parallel_info(int np) {
    printf("Number of MPI processes: %d\n", np);
    #pragma omp parallel
    {
        #pragma omp single
        printf("Number of OMP threads in each MPI process: %d\n", omp_get_num_threads());
    }
}

static void convert_to_double_parallel(uint8_t *image, double *imageD, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        imageD[i] = grayvalueToDouble(image[i], maxcolor);
    }
}

static void convert_from_double_parallel(double *imageD, uint8_t *image, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        image[i] = grayvalueFromDouble(imageD[i], maxcolor);
    }
}

static void write_output_image(const char* filename,
                               enum pnm_kind kind,
                               int rows,
                               int columns,
                               int maxcolor,
                               uint8_t *image) {
    if (ppp_pnm_write(filename, kind, rows, columns, maxcolor, image) == -1) {
        fprintf(stderr, "Could not write output to '%s'.\n", filename);
        
    }
}


void compute_parallel(const struct TaskInput *TI) {
    int self, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        print_parallel_info(np);
    }

    
    enum pnm_kind kind;
    int rows = 0, columns = 0, maxcolor = 0;
    uint8_t *full_image = NULL;
    double *full_imageD = NULL;

    if (self == 0) {
        full_image = load_and_validate_image(TI->filename, &kind, &rows, &columns, &maxcolor);
        full_imageD = (double *)malloc(rows * columns * sizeof(double));
        if (full_imageD == NULL) {
            fprintf(stderr, "Could not allocate memory for full imageD\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        convert_to_double_parallel(full_image, full_imageD, rows * columns, maxcolor);
    }

    
    broadcast_image_info(&rows, &columns, &maxcolor, &kind);

    
    int local_rows, global_start_row;
    calculate_row_distribution(rows, np, self, &local_rows, &global_start_row);

    
    int ghost_top    = (self > 0) ? 1 : 0;
    int ghost_bottom = (self < np - 1) ? 1 : 0;
    int total_alloc_rows = local_rows + ghost_top + ghost_bottom;

    
    double *local_imageD = NULL;
    double *local_tempD  = NULL;
    int ghost_rows = ghost_top + ghost_bottom;
    local_imageD = (double *)calloc((local_rows + ghost_rows) * columns, sizeof(double));
    local_tempD  = (double *)calloc((local_rows + ghost_rows) * columns, sizeof(double));
    if (local_imageD == NULL || local_tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for local arrays on rank %d\n", self);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {
        
        double *temp_recv = (double *)malloc(local_rows * columns * sizeof(double));
        if (temp_recv == NULL) {
            fprintf(stderr, "Could not allocate temp_recv on rank %d\n", self);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        
        distribute_image_data(full_imageD, temp_recv, rows, columns, np, self);

        
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < local_rows; y++) {
            for (int x = 0; x < columns; x++) {
                local_imageD[(y + ghost_top) * columns + x] = temp_recv[y * columns + x];
            }
        }

        free(temp_recv);
    }

    if (TI->doVCD) {
        vcd_parallel_v2(&local_imageD, &local_tempD,
                     local_rows, columns,
                     TI, self, np, global_start_row, rows);
    }


    if (TI->doSobel) {
        sobel_parallel(&local_imageD, &local_tempD,
                       local_rows, columns,
                       TI->sobelC, self, np);
    }

    uint8_t *full_image_out = NULL;
    double  *full_imageD_out = NULL;
    if (self == 0) {
        full_imageD_out = (double *)malloc(rows * columns * sizeof(double));
        if (full_imageD_out == NULL) {
            fprintf(stderr, "Could not allocate memory for full_imageD_out\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    
    {
        double *temp_send = (double *)malloc(local_rows * columns * sizeof(double));
        if (temp_send == NULL) {
            fprintf(stderr, "Could not allocate temp_send on rank %d\n", self);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < local_rows; y++) {
            for (int x = 0; x < columns; x++) {
                temp_send[y * columns + x] = local_imageD[(y + ghost_top) * columns + x];
            }
        }

        
        collect_image_data(temp_send, full_imageD_out,
                           rows, columns, np, self);

        free(temp_send);
    }

    
    if (self == 0) {
        full_image_out = (uint8_t *)malloc(rows * columns * sizeof(uint8_t));
        if (full_image_out == NULL) {
            fprintf(stderr, "Could not allocate full_image_out on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        convert_from_double_parallel(full_imageD_out, full_image_out, rows * columns, maxcolor);

        if (TI->outfilename != NULL) {
            write_output_image(TI->outfilename, kind, rows, columns, maxcolor, full_image_out);
        }

        free(full_image_out);
        free(full_imageD_out);
        free(full_image);
        free(full_imageD);
    }

    
    free(local_imageD);
    free(local_tempD);
}
