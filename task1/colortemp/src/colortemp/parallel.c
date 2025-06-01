#define _GNU_SOURCE

#include "parallel.h"

static int np;   // total number of MPI processes
static int self; // own process rank

static bool parallel_loading; // whether to use ppp_pnm_load_part
static bool parallel_saving;  // whether to save using MPI parallel I/O

uint8_t *partfn(enum pnm_kind kind, int rows, int columns, int *offset, int *length)
{
    int base_rows = rows / np;
    int extra_rows = rows % np;
    int num_rows = base_rows + (self < extra_rows ? 1 : 0);
    int start_row = self * base_rows + (self < extra_rows ? self : extra_rows);
    
    int bytes_per_pixel = (kind == PNM_KIND_PGM) ? 1 : 3;
    *offset = start_row * columns;
    *length = num_rows * columns;
    
    
    uint8_t *buffer = (uint8_t*)malloc(*length);
    if (!buffer) {
        fprintf(stderr, "Process %d: Failed to allocate %d bytes for image data\n", self, *length);
        return NULL;
    }
    
    printf("Process %d: partfn called - offset=%d, length=%d, rows=%d, start_row=%d, bufffer = %d\n", 
           self, *offset, *length, num_rows, start_row, buffer);

    return buffer;
}

void load_image_single(Image* img, const char* filename) {
    // Image *img = (Image*)malloc(sizeof(Image));
    img->image = ppp_pnm_read(filename, &img->kind, &img->rows, &img->columns, &img->maxval);
    if (img->image == NULL) {
        if (self == 0) {
            fprintf(stderr, "Could not load image from file '%s'.\n", filename);
        }
        free(img);
    }
    img->start_row = 0;
    img->num_rows = img->rows;
    img->estimated_global_temp = 0.0;
}

void distribute_image(Image* img) {
    if(self != 0) {
        return;
    }

    MPI_Bcast(&img->rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->maxval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->kind, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int base_rows = img->rows / np;
    int extra_rows = img->rows % np;
    
    MPI_Request *requests = malloc((np - 1) * sizeof(MPI_Request));
    for(int i = 1; i < np; i++){
        int num_rows = base_rows + (i < extra_rows ? 1 : 0);
        int start_row = i * base_rows + (i < extra_rows ? i : extra_rows);
        size_t offset = start_row * img->columns * 3;
        size_t size = num_rows * img->columns * 3;

        MPI_Isend(img->image + offset, size, MPI_BYTE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
    }
    MPI_Waitall(np - 1, requests, MPI_STATUS_IGNORE);
    free(requests);
    
    img->num_rows = base_rows + (0 < extra_rows ? 1 : 0);
    img->start_row = 0;
}

void recive_image(Image* img) {
    if(self == 0) {
        return;
    }

    MPI_Bcast(&img->rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->maxval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img->kind, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_rows = img->rows / np;
    int extra_rows = img->rows % np;

    img->num_rows = base_rows + (self < extra_rows ? 1 : 0);
    img->start_row = self * base_rows + (self < extra_rows ? self : extra_rows);
    
    size_t local_size = img->num_rows * img->columns * 3;
    img->local_size = local_size;
    
    img->image = (uint8_t *)malloc(local_size);
    if(img->image == NULL) {
        fprintf(stderr, "Process %d: Failed to allocate memory for image data\n", self);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    printf("Process %d: Receiving %zu bytes (%d rows) starting from row %d\n", 
           self, local_size, img->num_rows, img->start_row);
           
    MPI_Recv(img->image, local_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d: received image data, first pixel RGB: %d %d %d\n", 
           self, img->image[0], img->image[1], img->image[2]);
}

void load_image_part(Image* img, const char* filename) {
    img->image = ppp_pnm_read_part(filename, &img->kind, &img->rows, &img->columns, &img->maxval, partfn);
    
    if (img->image == NULL) {
        fprintf(stderr, "Process %d: Could not load image part from file '%s'.\n", self, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    int base_rows = img->rows / np;
    int extra_rows = img->rows % np;
    
    img->num_rows = base_rows + (self < extra_rows ? 1 : 0);
    img->start_row = self * base_rows + (self < extra_rows ? self : extra_rows);
    img->local_size = img->num_rows * img->columns * 3;  // Assuming RGB image (3 bytes per pixel)
    
    printf("Process %d: Loaded image part - rows=%d, columns=%d, start_row=%d, num_rows=%d\n", 
           self, img->rows, img->columns, img->start_row, img->num_rows);
           
    if (img->image) {
        printf("Process %d: First pixel RGB values: %d %d %d\n", 
               self, img->image[0], img->image[1], img->image[2]);
    }
    
    img->estimated_global_temp = 0.0;
}
void compute_collor_sum(Image* img) {
    long sum[3] = {0, 0, 0};
    for (int i = 0; i < img->num_rows * img->columns; ++i) {
        for (int j = 0; j <= 2; ++j) {
            sum[j] += img->image[3 * i + j];
        }
    }
    int pixels = img->num_rows * img->columns;
    double estimated_local_temp = rgbToColorTemp((double)sum[0] / pixels, (double)sum[1] / pixels, (double)sum[2] / pixels);
    printf("->Process %d: Estimated color temperature: %g K\n", self, estimated_local_temp);
    MPI_Allreduce(&estimated_local_temp, &img->estimated_global_temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void convert_color_temperature(Image* img, int from, int to) {
    // note: it should be more efficient just to do that in each process instead of sending additional data
    double from_r[3], to_r[3];
    colorTempToRGB(from, &from_r[0], &from_r[1], &from_r[2]);
    colorTempToRGB(to, &to_r[0], &to_r[1], &to_r[2]);
    if(self == 0) {
        printf("Adjusting color temperature from %d K to %d K.\n", from, to);
        printf("Original color temperature %d K corresponds to RGB color (%g, %g, %g).\n", from, from_r[0],
               from_r[1], from_r[2]);
        printf("New color temperature %d K corresponds to RGB color (%g, %g, %g).\n", to, to_r[0], to_r[1],
               to_r[2]);
    }
    for(int i = 0; i < img->num_rows * img->columns; ++i) {
        for(int j = 0; j <= 2; ++j) {
            img->image[3 * i + j] = rint(fmin(img->maxval, img->image[3 * i + j] * to_r[j] / from_r[j]));
        }
    }
}

void convert_color_temperature_shared(Image* img, int from, int to) {
    double from_r[3], to_r[3];
    colorTempToRGB(from, &from_r[0], &from_r[1], &from_r[2]);
    colorTempToRGB(to, &to_r[0], &to_r[1], &to_r[2]);
    
    if(self == 0) {
        printf("Adjusting color temperature from %d K to %d K.\n", from, to);
        printf("Original color temperature %d K corresponds to RGB color (%g, %g, %g).\n", 
               from, from_r[0], from_r[1], from_r[2]);
        printf("New color temperature %d K corresponds to RGB color (%g, %g, %g).\n", 
               to, to_r[0], to_r[1], to_r[2]);
    }    
    double ratio[3];
    for(int j = 0; j <= 2; ++j) {
        ratio[j] = to_r[j] / from_r[j];
    }    
    int total_pixels = img->num_rows * img->columns;
    #pragma omp parallel
    {
        #pragma omp single
        {
            if(self == 0) {
                printf("Process %d: Converting with %d OpenMP threads\n", self, omp_get_num_threads());
            }
        }        
        #pragma omp for schedule(static)
        for(int i = 0; i < total_pixels; ++i) {
            for(int j = 0; j <= 2; ++j) {
                img->image[3 * i + j] = rint(fmin(img->maxval, img->image[3 * i + j] * ratio[j]));
            }
        }
    }
    #pragma omp single
    {
        printf("Process %d: Color temperature conversion complete\n", self);
    }
}

void send_result_image(Image* img) {
    if(self == 0) {
        return;
    }
    MPI_Send(img->image, img->num_rows * img->columns * 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
}

void collect_result_image(Image* img) {
    if(self != 0) {
        return;
    }
    
    MPI_Request *requests = malloc((np - 1) * sizeof(MPI_Request));
    int base_rows = img->rows / np;
    int extra_rows = img->rows % np;

    for(int i = 1; i < np; i++) {
        int num_rows = base_rows + (i < extra_rows ? 1 : 0);
        int start_row = i * base_rows + (i < extra_rows ? i : extra_rows);
        size_t offset = start_row * img->columns * 3;
        
        printf("Process 0: receiving from proc %d, %d rows at offset %zu\n", 
               i, num_rows, offset / (img->columns * 3));
               
        MPI_Irecv(img->image + offset, num_rows * img->columns * 3, 
                 MPI_BYTE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
    }
    
    MPI_Waitall(np - 1, requests, MPI_STATUS_IGNORE);
    free(requests);
}

void compute_parallel(const struct TaskInput *TI) {
    double time_start, time_loaded, time_distributed;
    double time_reduced, time_processed, time_collected;
    double time_saved;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single
            printf("Number of OMP threads in each MPI process: %d\n", omp_get_num_threads());
        }
    }

    parallel_loading = TI->parallel_loading;
    parallel_saving = TI->parallel_saving;
    Image *img = (Image*)malloc(sizeof(Image));

    MPI_Barrier(MPI_COMM_WORLD);
    time_start = seconds();

    if(TI->parallel_loading) {
        // if(self != 0) return;
        // printf("parallel loading not implemented!\n");
        // return;
        load_image_part(img, TI->filename);
        printf("Process %d: loaded image part, first pixel RGB: %d %d %d\n", 
               self, img->image[0], img->image[1], img->image[2]);
    } else if(self == 0) {
        load_image_single(img, TI->filename);
        if (img == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
    }

    time_loaded = seconds();

    if(!TI->parallel_loading) {
        if(self==0) distribute_image(img);
        else recive_image(img);
    }

    time_distributed = seconds();

    // TODO: estimate color temperature of input image
    if(self!=0) compute_collor_sum(img);
    else MPI_Allreduce(&img->estimated_global_temp, &img->estimated_global_temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    img->estimated_global_temp /= (double)np;
    if (self == 0) {
        printf("%d, Estimated color temperature: %g K\n", self, img->estimated_global_temp);
    }

    time_reduced = seconds();

    // TODO: convert to new color temperature
    convert_color_temperature_shared(img, TI->originalColorTemp, TI->newColorTemp);

    time_processed = seconds();

    // TODO: collect output image (when not using parallel saving)
    if(!TI->parallel_saving) {
        if (self == 0) {
            collect_result_image(img);
        } else {
            send_result_image(img);
        }
        
    }
    time_collected = seconds();
    if (!TI->parallel_saving) {
        if (self == 0) {
            if (ppp_pnm_write(TI->outfilename, img->kind, img->rows, img->columns, img->maxval, img->image) != 0) {
                fprintf(stderr, "Could not write output image to %s\n", TI->outfilename);
            } else {
                printf("Process %d: Successfully wrote image to %s\n", self, TI->outfilename);
            }
        }
    } else {
        MPI_File file;
        int err = MPI_File_open(MPI_COMM_WORLD, TI->outfilename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        
        if (err != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(err, error_string, &length);
            fprintf(stderr, "Process %d: Error opening file %s: %s\n", self, TI->outfilename, error_string);
            MPI_Abort(MPI_COMM_WORLD, err);
            return;
        }
        
        char header[64];
        int header_len = 0;
        
        if (self == 0) {
            header_len = snprintf(header, sizeof(header), "P6\n%d %d\n%d\n", 
                                 img->columns, img->rows, img->maxval);
            
            err = MPI_File_write_at(file, 0, header, header_len, MPI_CHAR, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS) {
                char error_string[MPI_MAX_ERROR_STRING];
                int length;
                MPI_Error_string(err, error_string, &length);
                fprintf(stderr, "Process %d: Error writing header: %s\n", self, error_string);
                MPI_File_close(&file);
                MPI_Abort(MPI_COMM_WORLD, err);
                return;
            }
            printf("Process %d: Wrote header of length %d bytes\n", self, header_len);
        }
        
        MPI_Bcast(&header_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Offset offset = header_len + (MPI_Offset)(img->start_row * img->columns * 3);
        size_t data_size = img->num_rows * img->columns * 3;
        
        printf("Process %d: Writing %zu bytes at offset %lld\n", self, data_size, (long long)offset);
        
        err = MPI_File_write_at(file, offset, img->image, data_size, MPI_BYTE, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(err, error_string, &length);
            fprintf(stderr, "Process %d: Error writing image data: %s\n", self, error_string);
        }
        MPI_File_close(&file);
        
        if (self == 0) {
            printf("Parallel image write to %s complete\n", TI->outfilename);
        }
    }

    if(self==0) printf("immage info: %d %d %d %d\n", img->kind, img->rows, img->columns, img->rows*img->columns);

    time_saved = seconds();

    if (self == 0) {
        printf("Times:\n"
               "  Loading:       %.6f s\n"
               "  Distributing:  %.6f s\n"
               "  Reduction:     %.6f s\n"
               "  Processing:    %.6f s\n"
               "  Collecting:    %.6f s\n"
               "  Saving:        %.6f s\n"
               "  TOTAL:         %.6f s\n",
               time_loaded - time_start, time_distributed - time_loaded, time_reduced - time_distributed,
               time_processed - time_reduced, time_collected - time_processed, time_saved - time_collected,
               time_saved - time_start);
    }
}
