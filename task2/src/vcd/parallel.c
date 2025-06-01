#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

/*
 * Convert a gray value in the range 0..maxcolor to a double value in [0,1].
 */
inline static double grayvalueToDouble(uint8_t v, int maxcolor) {
    return (double)v / maxcolor;
}

/*
 * Convert a double value back to a gray value in the range 0..maxcolor.
 */
inline static int grayvalueFromDouble(double d, int maxcolor) {
    int v = lrint(d * maxcolor);
    return (v < 0 ? 0 : (v > maxcolor ? maxcolor : v));
}

/*
 * Swap the values pointed to by 'a' and 'b'
 */
static void swap(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

/*
 * Compute phi function for VCD algorithm
 */
static inline double compute_phi(double nu, double kappa) {
    const double chi = nu / kappa;
    return chi * exp(-chi * chi / 2.0);
}

/*
 * Compute xi function for VCD algorithm
 */
static inline double compute_xi(double nu, double kappa) {
    const double psi = nu / (kappa * sqrt(2.0));
    return 1 / sqrt(2.0) * psi * exp(-psi * psi / 2.0);
}

/*
 * Safe pixel access with boundary checking
 */
static inline double get_pixel_safe(double *image, int x, int y, int rows, int columns) {
    return (y >= 0 && y < rows && x >= 0 && x < columns) ? image[y * columns + x] : 0.0;
}

/*
 * Compute VCD delta for a single pixel
 */
static double compute_vcd_delta_for_pixel(double *image, int x, int y, int rows, int columns, double kappa) {
    double current = get_pixel_safe(image, x, y, rows, columns);
    
    // Compute directional differences
    double diff_right = get_pixel_safe(image, x + 1, y, rows, columns) - current;
    double diff_left = current - get_pixel_safe(image, x - 1, y, rows, columns);
    double diff_down = get_pixel_safe(image, x, y + 1, rows, columns) - current;
    double diff_up = current - get_pixel_safe(image, x, y - 1, rows, columns);
    
    // Diagonal differences
    double diff_diag1 = get_pixel_safe(image, x + 1, y + 1, rows, columns) - current;
    double diff_diag2 = current - get_pixel_safe(image, x - 1, y - 1, rows, columns);
    double diff_diag3 = get_pixel_safe(image, x - 1, y + 1, rows, columns) - current;
    double diff_diag4 = current - get_pixel_safe(image, x + 1, y - 1, rows, columns);
    
    // Apply phi and xi functions
    return compute_phi(diff_right, kappa) - compute_phi(diff_left, kappa) +
           compute_phi(diff_down, kappa) - compute_phi(diff_up, kappa) +
           compute_xi(diff_diag1, kappa) - compute_xi(diff_diag2, kappa) +
           compute_xi(diff_diag3, kappa) - compute_xi(diff_diag4, kappa);
}

/*
 * Perform one VCD iteration in parallel
 */
static double perform_vcd_iteration(double **image, double **temp, int rows, int columns, 
                                  double kappa, double dt) {
    double local_deltaMax = 0;

    #pragma omp parallel for collapse(2) reduction(max:local_deltaMax) schedule(static)
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            double delta = compute_vcd_delta_for_pixel(*image, x, y, rows, columns, kappa);
            (*temp)[y * columns + x] = (*image)[y * columns + x] + kappa * dt * delta;
            
            // Only consider interior pixels for deltaMax calculation
            if (y > 0 && y < rows - 1 && x > 0 && x < columns - 1) {
                delta = fabs(delta);
                if (delta > local_deltaMax)
                    local_deltaMax = delta;
            }
        }
    }
    
    swap(image, temp);
    return local_deltaMax;
}

/*
 * Check if VCD algorithm should continue
 */
static int should_continue_vcd(int iteration, int max_iterations, double deltaMax, double epsilon) {
    return iteration < max_iterations && deltaMax > epsilon;
}

/*
 * Parallel VCD operator using OpenMP
 */
void vcd_parallel(double **image, double **temp, int rows, const int columns, const struct TaskInput *TI) {
    const double kappa = TI->vcdKappa;
    const double epsilon = TI->vcdEpsilon;
    const double dt = TI->vcdDt;
    const int N = TI->vcdN;

    int iteration = 0;
    double deltaMax = epsilon + 1.0;
    
    while (should_continue_vcd(iteration, N, deltaMax, epsilon)) {
        iteration++;
        deltaMax = perform_vcd_iteration(image, temp, rows, columns, kappa, dt);
        
        if (TI->debugOutput)
            printf("Iteration %2d: max. Delta = %g\n", iteration, deltaMax);
    }
}

/*
 * Compute Sobel gradient components for a single pixel
 */
static void compute_sobel_gradients(double *image, int x, int y, int rows, int columns, 
                                  double *sx, double *sy) {
    // Compute sx (horizontal gradient)
    *sx = get_pixel_safe(image, x - 1, y - 1, rows, columns) + 
          2 * get_pixel_safe(image, x, y - 1, rows, columns) + 
          get_pixel_safe(image, x + 1, y - 1, rows, columns) - 
          get_pixel_safe(image, x - 1, y + 1, rows, columns) - 
          2 * get_pixel_safe(image, x, y + 1, rows, columns) - 
          get_pixel_safe(image, x + 1, y + 1, rows, columns);
    
    // Compute sy (vertical gradient)
    *sy = get_pixel_safe(image, x - 1, y - 1, rows, columns) + 
          2 * get_pixel_safe(image, x - 1, y, rows, columns) + 
          get_pixel_safe(image, x - 1, y + 1, rows, columns) - 
          get_pixel_safe(image, x + 1, y - 1, rows, columns) - 
          2 * get_pixel_safe(image, x + 1, y, rows, columns) - 
          get_pixel_safe(image, x + 1, y + 1, rows, columns);
}

/*
 * Apply Sobel operator to all pixels in parallel
 */
static void apply_sobel_operator(double **input, double **temp, int rows, int columns, double sobelC) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
            double sx, sy;
            compute_sobel_gradients(*input, x, y, rows, columns, &sx, &sy);
            (*temp)[y * columns + x] = sobelC * hypot(sx, sy);
        }
    }
    
    swap(input, temp);
}

/*
 * Parallel Sobel Operator using OpenMP
 */
void sobel_parallel(double **input, double **temp, int rows, const int columns, double sobelC) {
    apply_sobel_operator(input, temp, rows, columns, sobelC);
}

/*
 * Load and validate image file
 */
static uint8_t* load_and_validate_image(const char* filename, enum pnm_kind *kind, 
                                       int *rows, int *columns, int *maxcolor) {
    uint8_t *image = ppp_pnm_read(filename, kind, rows, columns, maxcolor);

    if (image == NULL) {
        fprintf(stderr, "Could not load image from file '%s'.\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else if (*kind != PNM_KIND_PGM) {
        fprintf(stderr, "Image is not a \"portable graymap.\"\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    return image;
}

/*
 * Allocate memory for double precision image arrays
 */
static void allocate_double_arrays(double **imageD, double **tempD, int size) {
    *imageD = (double *)malloc(sizeof(double) * size);
    *tempD = (double *)malloc(sizeof(double) * size);
    
    if (*imageD == NULL || *tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for the image\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/*
 * Convert uint8 image to double precision in parallel
 */
static void convert_to_double_parallel(uint8_t *image, double *imageD, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        imageD[i] = grayvalueToDouble(image[i], maxcolor);
    }
}

/*
 * Convert double precision image back to uint8 in parallel
 */
static void convert_from_double_parallel(double *imageD, uint8_t *image, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        image[i] = grayvalueFromDouble(imageD[i], maxcolor);
    }
}

/*
 * Apply image processing algorithms
 */
static void apply_image_processing(double **imageD, double **tempD, int rows, int columns, 
                                 const struct TaskInput *TI) {
    if (TI->doVCD) {
        if (TI->improvedVCD) {
            // For now, use the same parallel VCD - could be enhanced further
            vcd_parallel(imageD, tempD, rows, columns, TI);
        } else {
            vcd_parallel(imageD, tempD, rows, columns, TI);
        }
    }

    if (TI->doSobel) {
        sobel_parallel(imageD, tempD, rows, columns, TI->sobelC);
    }
}

/*
 * Write output image to file
 */
static void write_output_image(const char* filename, enum pnm_kind kind, int rows, 
                             int columns, int maxcolor, uint8_t *image) {
    if (ppp_pnm_write(filename, kind, rows, columns, maxcolor, image) == -1) {
        fprintf(stderr, "Could not write output to '%s'.\n", filename);
    }
}

/*
 * Print MPI and OpenMP configuration
 */
static void print_parallel_info(int np) {
    printf("Number of MPI processes: %d\n", np);
    #pragma omp parallel
    {
        #pragma omp single
        printf("Number of OMP threads in each MPI process: %d\n", omp_get_num_threads());
    }
}

/*
 * Structure to hold MPI distribution information
 */
typedef struct {
    int local_rows;          // Number of rows assigned to this process
    int start_row;           // Starting row index (global)
    int end_row;             // Ending row index (global, exclusive)
    int local_rows_with_halo; // Local rows including halo regions
    int halo_start_row;      // Starting row including halo (global)
    int halo_end_row;        // Ending row including halo (global, exclusive)
} MPIDistribution;

/*
 * Calculate row distribution for MPI processes
 */
static MPIDistribution calculate_mpi_distribution(int total_rows, int rank, int size) {
    MPIDistribution dist;
    
    // Basic row distribution
    int base_rows = total_rows / size;
    int remainder = total_rows % size;
    
    dist.local_rows = base_rows + (rank < remainder ? 1 : 0);
    dist.start_row = rank * base_rows + (rank < remainder ? rank : remainder);
    dist.end_row = dist.start_row + dist.local_rows;
    
    // Calculate halo regions (1 row above and below for stencil operations)
    dist.halo_start_row = (dist.start_row > 0) ? dist.start_row - 1 : dist.start_row;
    dist.halo_end_row = (dist.end_row < total_rows) ? dist.end_row + 1 : dist.end_row;
    dist.local_rows_with_halo = dist.halo_end_row - dist.halo_start_row;
    
    return dist;
}

/*
 * Distribute image data from process 0 to all processes with halo regions
 */
static void distribute_image_data(double *full_image, double **local_image, 
                                int total_rows, int columns, int rank, int size,
                                MPIDistribution *dist) {
    if (rank == 0) {
        // Process 0: send data to other processes
        for (int p = 1; p < size; p++) {
            MPIDistribution p_dist = calculate_mpi_distribution(total_rows, p, size);
            int send_size = p_dist.local_rows_with_halo * columns;
            int start_idx = p_dist.halo_start_row * columns;
            
            MPI_Send(&full_image[start_idx], send_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        
        // Process 0: copy its own data
        int local_size = dist->local_rows_with_halo * columns;
        *local_image = (double*)malloc(sizeof(double) * local_size);
        memcpy(*local_image, &full_image[dist->halo_start_row * columns], 
               sizeof(double) * local_size);
    } else {
        // Other processes: receive data
        int local_size = dist->local_rows_with_halo * columns;
        *local_image = (double*)malloc(sizeof(double) * local_size);
        
        MPI_Recv(*local_image, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/*
 * Collect processed image data from all processes back to process 0
 */
static void collect_image_data(double *local_image, double **full_image,
                             int total_rows, int columns, int rank, int size,
                             MPIDistribution *dist) {
    if (rank == 0) {
        // Process 0: copy its own data (excluding halo)
        int local_start_offset = (dist->start_row - dist->halo_start_row) * columns;
        memcpy(&(*full_image)[dist->start_row * columns], 
               &local_image[local_start_offset], 
               sizeof(double) * dist->local_rows * columns);
        
        // Process 0: receive data from other processes
        for (int p = 1; p < size; p++) {
            MPIDistribution p_dist = calculate_mpi_distribution(total_rows, p, size);
            int recv_size = p_dist.local_rows * columns;
            
            MPI_Recv(&(*full_image)[p_dist.start_row * columns], recv_size, MPI_DOUBLE, 
                    p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Other processes: send their data (excluding halo)
        int local_start_offset = (dist->start_row - dist->halo_start_row) * columns;
        int send_size = dist->local_rows * columns;
        
        MPI_Send(&local_image[local_start_offset], send_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
}

/*
 * Exchange halo data between neighboring MPI processes
 */
static void exchange_halo_data(double *local_image, int columns, int rank, int size,
                             MPIDistribution *dist, int total_rows) {
    // Send/receive halo rows with neighbors
    MPI_Request requests[4];
    int req_count = 0;
    
    // Exchange with upper neighbor (rank - 1)
    if (rank > 0 && dist->start_row > 0) {
        // Send top interior row to upper neighbor
        int send_row_idx = (dist->start_row - dist->halo_start_row) * columns;
        MPI_Isend(&local_image[send_row_idx], columns, MPI_DOUBLE, 
                 rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        
        // Receive halo row from upper neighbor
        int recv_row_idx = 0; // First row is halo from upper neighbor
        MPI_Irecv(&local_image[recv_row_idx], columns, MPI_DOUBLE, 
                 rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Exchange with lower neighbor (rank + 1)
    if (rank < size - 1 && dist->end_row < total_rows) {
        // Send bottom interior row to lower neighbor
        int send_row_idx = (dist->end_row - 1 - dist->halo_start_row) * columns;
        MPI_Isend(&local_image[send_row_idx], columns, MPI_DOUBLE, 
                 rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        // Receive halo row from lower neighbor
        int recv_row_idx = (dist->local_rows_with_halo - 1) * columns;
        MPI_Irecv(&local_image[recv_row_idx], columns, MPI_DOUBLE, 
                 rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Wait for all communications to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
}

/*
 * MPI-parallel VCD implementation
 */
static void vcd_mpi_parallel(double **local_image, double **local_temp, 
                           int columns, int rank, int size, int total_rows,
                           MPIDistribution *dist, const struct TaskInput *TI) {
    const double kappa = TI->vcdKappa;
    const double epsilon = TI->vcdEpsilon;
    const double dt = TI->vcdDt;
    const int N = TI->vcdN;

    int iteration = 0;
    double global_deltaMax = epsilon + 1.0;
    
    while (should_continue_vcd(iteration, N, global_deltaMax, epsilon)) {
        iteration++;
        
        // Exchange halo data with neighbors
        exchange_halo_data(*local_image, columns, rank, size, dist, total_rows);
        
        double local_deltaMax = 0;
        
        // Process only the interior rows (excluding halo)
        int start_y = (dist->start_row > 0) ? 1 : 0; // Skip halo if exists
        int end_y = dist->local_rows_with_halo - ((dist->end_row < total_rows) ? 1 : 0);
        
        #pragma omp parallel for reduction(max:local_deltaMax) schedule(static)
        for (int local_y = start_y; local_y < end_y; local_y++) {
            for (int x = 0; x < columns; x++) {
                int global_y = dist->halo_start_row + local_y;
                double delta = compute_vcd_delta_for_pixel(*local_image, x, local_y, 
                                                        dist->local_rows_with_halo, columns, kappa);
                (*local_temp)[local_y * columns + x] = (*local_image)[local_y * columns + x] + kappa * dt * delta;
                
                // Only consider interior pixels for deltaMax calculation
                if (global_y > 0 && global_y < total_rows - 1 && x > 0 && x < columns - 1) {
                    delta = fabs(delta);
                    if (delta > local_deltaMax)
                        local_deltaMax = delta;
                }
            }
        }
        
        // Find global maximum delta across all processes
        MPI_Allreduce(&local_deltaMax, &global_deltaMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        swap(local_image, local_temp);
        
        if (TI->debugOutput && rank == 0)
            printf("Iteration %2d: max. Delta = %g\n", iteration, global_deltaMax);
    }
}

/*
 * MPI-parallel Sobel implementation
 */
static void sobel_mpi_parallel(double **local_image, double **local_temp, 
                             int columns, int rank, int size, int total_rows,
                             MPIDistribution *dist, double sobelC) {
    // Exchange halo data with neighbors
    exchange_halo_data(*local_image, columns, rank, size, dist, total_rows);
    
    // Process only the interior rows (excluding halo)
    int start_y = (dist->start_row > 0) ? 1 : 0; // Skip halo if exists
    int end_y = dist->local_rows_with_halo - ((dist->end_row < total_rows) ? 1 : 0);
    
    #pragma omp parallel for schedule(static)
    for (int local_y = start_y; local_y < end_y; local_y++) {
        for (int x = 0; x < columns; x++) {
            double sx, sy;
            compute_sobel_gradients(*local_image, x, local_y, 
                                  dist->local_rows_with_halo, columns, &sx, &sy);
            (*local_temp)[local_y * columns + x] = sobelC * hypot(sx, sy);
        }
    }
    
    swap(local_image, local_temp);
}

/*
 * Apply MPI-parallel image processing
 */
static void apply_mpi_image_processing(double **local_image, double **local_temp,
                                     int columns, int rank, int size, int total_rows,
                                     MPIDistribution *dist, const struct TaskInput *TI) {
    if (TI->doVCD) {
        if (TI->improvedVCD) {
            // For now, use the same parallel VCD - could be enhanced further
            vcd_mpi_parallel(local_image, local_temp, columns, rank, size, 
                           total_rows, dist, TI);
        } else {
            vcd_mpi_parallel(local_image, local_temp, columns, rank, size, 
                           total_rows, dist, TI);
        }
    }

    if (TI->doSobel) {
        sobel_mpi_parallel(local_image, local_temp, columns, rank, size, 
                         total_rows, dist, TI->sobelC);
    }
}

/*
 * Main parallel computation function with MPI distribution
 */
void compute_parallel(const struct TaskInput *TI) {
    int self, np;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        print_parallel_info(np);
    }

    // Load image and broadcast dimensions
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = NULL;
    double *full_imageD = NULL;
    
    if (self == 0) {
        image = load_and_validate_image(TI->filename, &kind, &rows, &columns, &maxcolor);
        
        int image_size = rows * columns;
        allocate_double_arrays(&full_imageD, NULL, image_size);
        convert_to_double_parallel(image, full_imageD, image_size, maxcolor);
    }
    
    // Broadcast image dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kind, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate MPI distribution
    MPIDistribution dist = calculate_mpi_distribution(rows, self, np);
    
    // Distribute image data with halo regions
    double *local_imageD = NULL;
    distribute_image_data(full_imageD, &local_imageD, rows, columns, self, np, &dist);
    
    // Allocate local temporary buffer
    double *local_tempD = (double*)malloc(sizeof(double) * dist.local_rows_with_halo * columns);
    if (local_tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for local temporary buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double time_loaded = seconds();

    // Apply image processing algorithms with MPI parallelism
    apply_mpi_image_processing(&local_imageD, &local_tempD, columns, self, np, 
                             rows, &dist, TI);

    double time_computed = seconds();

    // Collect results back to process 0
    collect_image_data(local_imageD, &full_imageD, rows, columns, self, np, &dist);

    // Convert back and write output on process 0
    if (self == 0) {
        int image_size = rows * columns;
        convert_from_double_parallel(full_imageD, image, image_size, maxcolor);

        if (TI->outfilename != NULL) {
            write_output_image(TI->outfilename, kind, rows, columns, maxcolor, image);
        }

        printf("Computation time: %.6f\n", time_computed - time_loaded);

        // Cleanup
        free(full_imageD);
        free(image);
    }
    
    // Cleanup local data
    free(local_imageD);
    free(local_tempD);
}