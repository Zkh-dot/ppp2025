#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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
 * Main parallel computation function
 */
void compute_parallel(const struct TaskInput *TI) {
    int self, np;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        print_parallel_info(np);
    }

    // Load image only on process 0 for now (full parallel implementation would distribute this)
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image = NULL;
    double *imageD = NULL;
    double *tempD = NULL;

    if (self == 0) {
        image = load_and_validate_image(TI->filename, &kind, &rows, &columns, &maxcolor);
        
        int image_size = rows * columns;
        allocate_double_arrays(&imageD, &tempD, image_size);
        convert_to_double_parallel(image, imageD, image_size, maxcolor);
    }

    double time_loaded = seconds();

    if (self == 0) {
        apply_image_processing(&imageD, &tempD, rows, columns, TI);
    }

    double time_computed = seconds();

    // Convert back and write output on process 0
    if (self == 0) {
        int image_size = rows * columns;
        convert_from_double_parallel(imageD, image, image_size, maxcolor);

        if (TI->outfilename != NULL) {
            write_output_image(TI->outfilename, kind, rows, columns, maxcolor, image);
        }

        printf("Computation time: %.6f\n", time_computed - time_loaded);

        // Cleanup
        free(imageD);
        free(tempD);
        free(image);
    }
}