// #ifndef PARALLEL_H
// #define PARALLEL_H


#include <math.h>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

#include <stdbool.h>
#include <stdint.h>
#include "ppp_pnm/ppp_pnm.h"

typedef struct {
    double from[3];
    double to[3];
} conv_data;

typedef struct {
    enum pnm_kind kind;    
    int rows;              
    int columns;           
    int maxval;            
    uint8_t *image;        
    int start_row;         
    int num_rows;          
    int local_size;
    double estimated_global_temp;
} Image;

void compute_parallel(const struct TaskInput *TI);

void load_image_single(Image* img, const char* filename);

void load_image_part(Image* img, const char* filename);

uint8_t *partfn(enum pnm_kind kind, int rows, int columns, int *offset, int *length);

void distribute_image(Image* img);

void recive_image(Image* img);

void convert_color_temperature_shared(Image* img, int from, int to);

void convert_color_temperature(Image* img, int from, int to);

void collect_result_image(Image* img);

void send_result_image(Image* img);

void save_image_single(Image* img, const char* filename);

void save_image_parallel(Image* local_img, int total_rows, int total_columns, const char* filename);

void free_image(Image* img);

// #endif /* PARALLEL_H */
