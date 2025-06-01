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
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

/*
 * Broadcast image metadata (rows, columns, maxcolor, kind) from rank 0 to all.
 */
static void broadcast_image_info(int *rows, int *columns, int *maxcolor, enum pnm_kind *kind) {
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kind, sizeof(enum pnm_kind), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/*
 * Determine, сколько строк (без учёта ghost) получает каждый процесс,
 * и с какого глобального индекса эти строки начинаются.
 */
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

/*
 * Distribute image data (double-значения, уже готовые в full_imageD) 
 * от процесса 0 ко всем. Каждый процесс получит ровно local_rows*columns элементов.
 */
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

/*
 * Collect image data (double-значения) от всех процессов в full_image
 * на процессе 0. Каждый процесс отсылает local_rows*columns элементов,
 * соответствующие своему «кусочку» (без учёта ghost).
 */
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

/*
 * Load image (uint8_t) и проверка, что она типа PGM. Если ошибка — Abort.
 */
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

/*
 * Обмен «ghost rows» между соседними процессами:
 * - Если есть верхний сосед (rank>0), то process отправляет свою первую «реальную» строку
 *   (которая расположена в local_data[columns], т.к. row=0 зарезервирована под ghost-top)
 *   и принимает в local_data[0] (ghost-top).
 * - Если есть нижний сосед (rank<np-1), то отправляет последнюю «реальную» строку
 *   (индекс (local_rows-1)+ghost_top) и принимает в последнюю ghost-bottom (local_rows+ghost_top).
 */
static void exchange_ghost_rows(double *local_data, int local_rows, int columns,
                                int rank, int np) {
    MPI_Request requests[4];
    int req_count = 0;

    int ghost_top = (rank > 0) ? 1 : 0;
    int ghost_bottom = (rank < np - 1) ? 1 : 0;

    // Если есть верхний сосед
    if (rank > 0) {
        // Отправляем первую реальную строку (y=0 ⇒ в массиве это строка index = ghost_top =1)
        MPI_Isend(&local_data[ghost_top * columns], columns, MPI_DOUBLE,
                  rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        // Принимаем его «нижнюю» строку в local_data[0]
        MPI_Irecv(&local_data[0], columns, MPI_DOUBLE,
                  rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Если есть нижний сосед
    if (rank < np - 1) {
        // Отправляем последнюю реальную строку (y=local_rows-1 ⇒ в массиве index = (local_rows-1)+ghost_top)
        MPI_Isend(&local_data[(local_rows - 1 + ghost_top) * columns], columns, MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        // Принимаем его «верхнюю» строку в local_data[(local_rows + ghost_top)]
        MPI_Irecv(&local_data[(local_rows + ghost_top) * columns], columns, MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Ждём завершения всех отправок/приёмов
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

/*
 * Доступ к пикселю с учётом «ghost» (призрачных) строк.
 * Локальный массив построен так: 
 *   total_allocated_rows = local_rows + ghost_top + ghost_bottom;
 *   строки с индексом 0 и total_allocated_rows-1 (если ghost_top/bottom==1) — это ghost.
 * Если x или y «вне диапазона» (вне [0,columns-1] или вне [0, local_rows-1] без offset),
 * возвращаем 0. Иначе возвращаем local_data[(y+ghost_top)*columns + x].
 */
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

/*
 * Функции phi и xi из single.c:
 */
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

/*
 * Вычислить «дельту» для одного пикселя VCD (распределённая версия).
 * Здесь image — это указатель на локальный буфер (с ghost-строками),
 * x∈[0..columns-1], y∈[0..local_rows-1] (без учёта offset),
 * а ghost_top/bottom = 1 или 0 в зависимости от имеющихся соседей.
 */
static double compute_vcd_delta_for_pixel(double *image,
                                          int x, int y,
                                          int local_rows, int columns,
                                          double kappa,
                                          int ghost_top, int ghost_bottom) {
    double current = get_pixel_safe(image, x, y, local_rows, columns, ghost_top, ghost_bottom);

    // направленные разности
    double diff_right = get_pixel_safe(image, x + 1, y, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_left  = current - get_pixel_safe(image, x - 1, y, local_rows, columns, ghost_top, ghost_bottom);
    double diff_down  = get_pixel_safe(image, x, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_up    = current - get_pixel_safe(image, x, y - 1, local_rows, columns, ghost_top, ghost_bottom);

    // диагональные разности
    double diff_diag1 = get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_diag2 = current - get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom);
    double diff_diag3 = get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom) - current;
    double diff_diag4 = current - get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom);

    // применяем phi и xi, как в single.c:
    return phi(diff_right) - phi(diff_left)
         + phi(diff_down)  - phi(diff_up)
         + xi(diff_diag1)  - xi(diff_diag2)
         + xi(diff_diag3)  - xi(diff_diag4);
}

/*
 * Выполнить одну итерацию VCD для всего локального изображения (distributed).
 * Возвращает локальный максимум |delta|, который затем сведётся в глобальный через MPI_Allreduce.
 */
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
    // Обмен ghost rows перед вычислением
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

            // Обновляем локальный максимум |delta| только для тех пикселей,
            // которые в глобальном масштабе не на граничных строках и столбцах
            int global_y = global_start_row + y;
            if (global_y > 0 && global_y < total_rows - 1 && x > 0 && x < columns - 1) {
                double abs_delta = fabs(delta);
                if (abs_delta > local_deltaMax) {
                    local_deltaMax = abs_delta;
                }
            }
        }
    }

    // Меняем указатели, чтобы следующий раз работать с уже обновлённым изображением:
    swap(image, temp);
    return local_deltaMax;
}

/*
 * Полный параллельный VCD (MPI + OpenMP).
 * Каждый процесс выполняет N итераций (или до сходимости eps),
 * а затем меняет local_image и local_temp местами.
 */
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

        // Сводим максимум δ по всем процессам
        MPI_Allreduce(&local_deltaMax, &global_deltaMax,
                      1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (TI->debugOutput && rank == 0) {
            printf("VCD Iteration %2d: max |Δ| = %g\n", iteration, global_deltaMax);
        }
    }
}

/*
 * Вычислить горизонтальный (sx) и вертикальный (sy) градиенты Sobel
 * для одного пикселя (x,y) из локального буфера image (с ghost).
 */
static void compute_sobel_gradients_distributed(double *image,
                                                int x, int y,
                                                int local_rows,
                                                int columns,
                                                double *sx,
                                                double *sy,
                                                int ghost_top,
                                                int ghost_bottom) {
    // Горизонтальный градиент (sx)
    *sx =  get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         + 2 * get_pixel_safe(image, x    , y - 1, local_rows, columns, ghost_top, ghost_bottom)
         +   get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom)
         - 2 * get_pixel_safe(image, x    , y + 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom);

    // Вертикальный градиент (sy)
    *sy =  get_pixel_safe(image, x - 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         + 2 * get_pixel_safe(image, x - 1, y    , local_rows, columns, ghost_top, ghost_bottom)
         +   get_pixel_safe(image, x - 1, y + 1, local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y - 1, local_rows, columns, ghost_top, ghost_bottom)
         - 2 * get_pixel_safe(image, x + 1, y    , local_rows, columns, ghost_top, ghost_bottom)
         -   get_pixel_safe(image, x + 1, y + 1, local_rows, columns, ghost_top, ghost_bottom);
}

/*
 * Применить оператор Собеля к локальному фрагменту (MPI + OpenMP).
 * Результат кладём в temp (double-матрицу с теми же ghost).
 */
static void apply_sobel_operator_distributed(double **input,
                                             double **temp,
                                             int local_rows,
                                             int columns,
                                             double sobelC,
                                             int rank,
                                             int np) {
    // Сначала нужно обменяться ghost-строками у входного изображения
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

    // Свапим указатели, чтобы в конце (*input) содержало результат
    swap(input, temp);
}

/*
 * Оболочка над apply_sobel_operator_distributed, чтобы интерфейс был похож на single.c
 */
void sobel_parallel(double **input, double **temp,
                    int local_rows, int columns,
                    double sobelC, int rank, int np) {
    apply_sobel_operator_distributed(input, temp, local_rows, columns, sobelC, rank, np);
}

/*
 * Распечатать информацию о количестве MPI-процессов и потоков OpenMP (только rank=0).
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
 * Параллельная конвертация uint8 → double
 */
static void convert_to_double_parallel(uint8_t *image, double *imageD, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        imageD[i] = grayvalueToDouble(image[i], maxcolor);
    }
}

/*
 * Параллельная конвертация double → uint8
 */
static void convert_from_double_parallel(double *imageD, uint8_t *image, int size, int maxcolor) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        image[i] = grayvalueFromDouble(imageD[i], maxcolor);
    }
}

/*
 * Записать PGM-наблюдаемое в файл (тип PNM_KIND_PGM)
 */
static void write_output_image(const char* filename,
                               enum pnm_kind kind,
                               int rows,
                               int columns,
                               int maxcolor,
                               uint8_t *image) {
    if (ppp_pnm_write(filename, kind, rows, columns, maxcolor, image) == -1) {
        fprintf(stderr, "Could not write output to '%s'.\n", filename);
        // мы не делаем MPI_Abort(), пусть все процессы завершаются корректно
    }
}

/*
 * Основная функция для параллельного вычисления.
 * Здесь мы:
 *  1) Инициализируем MPI, узнаём rank,np
 *  2) На rank=0: загружаем изображение → uint8, конвертим в double
 *  3) Рассылаем metadata всем через broadcast
 *  4) Распределяем фрагменты full_imageD по MPI-запросу (Scatterv)
 *  5) В каждом процессе у нас локальный буфер local_imageD (с ghost-строками).
 *     Нам нужно учесть, что фактически local_data выделен больший, чем local_rows*columns:
 *       total_allocated_rows = local_rows + ghost_top + ghost_bottom
 *     Поэтому local_imageD points на сдвиг ghost_top (raw buffer).
 *  6) Применяем VCD (если надо) и/или Sobel параллельно.
 *  7) Собираем результат с учётом ghost-offset (Collect + Convert→uint8)
 *  8) На rank=0: пишем файл, освобождаем память.
 *  9) cleanup и выход.
 */
void compute_parallel(const struct TaskInput *TI) {
    int self, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        print_parallel_info(np);
    }

    // 1) На rank=0: загрузка image (uint8) + metadata
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
        // Конвертируем uint8→double
        convert_to_double_parallel(full_image, full_imageD, rows * columns, maxcolor);
    }

    // 2) Рассылаем всем процессам размеры rows, columns, maxcolor, kind
    broadcast_image_info(&rows, &columns, &maxcolor, &kind);

    // 3) Считаем, сколько строк (без ghost) получает каждый процесс, и с какого start_row одной полосы
    int local_rows, global_start_row;
    calculate_row_distribution(rows, np, self, &local_rows, &global_start_row);

    // 4) Определяем, сколько ghost-строк нужно (топ и боттом)
    int ghost_top    = (self > 0) ? 1 : 0;
    int ghost_bottom = (self < np - 1) ? 1 : 0;
    int total_alloc_rows = local_rows + ghost_top + ghost_bottom;

    // 5) Выделяем локальные буферы под double-данные (с учётом ghost)
    double *local_imageD = NULL;
    double *local_tempD  = NULL;
    int ghost_rows = ghost_top + ghost_bottom;
    local_imageD = (double *)calloc((local_rows + ghost_rows) * columns, sizeof(double));
    local_tempD  = (double *)calloc((local_rows + ghost_rows) * columns, sizeof(double));
    if (local_imageD == NULL || local_tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for local arrays on rank %d\n", self);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
     * 6) Распределяем данные full_imageD (double-значения) по процессам.
     *    Однако функция Scatterv работает только с непрерывным буфером без «пробелов».
     *    Поэтому нам нужно в каждом процессе скопировать свои «local_rows*columns» значений
     *    в диапазон local_imageD[ghost_top*columns … (ghost_top+local_rows-1)*columns] 
     *    (вне ghost).
     */
    {
        // временный буфер для приёма без учёта ghost
        double *temp_recv = (double *)malloc(local_rows * columns * sizeof(double));
        if (temp_recv == NULL) {
            fprintf(stderr, "Could not allocate temp_recv on rank %d\n", self);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Рассылаем фрагмент изображения
        distribute_image_data(full_imageD, temp_recv, rows, columns, np, self);

        // Заполняем local_imageD так, чтобы учесть ghost_top
        // Копируем каждую строку: temp_recv[y*columns + x] → local_imageD[(y+ghost_top)*columns + x]
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < local_rows; y++) {
            for (int x = 0; x < columns; x++) {
                local_imageD[(y + ghost_top) * columns + x] = temp_recv[y * columns + x];
            }
        }

        free(temp_recv);
    }

    /*
     * 7) Запускаем VCD-процедуру (MPI+OpenMP) или нет:
     *     В single.c наверняка есть флаг TI->doVcd (либо похожий), 
     *     проверяем его и запускаем:
     */
    if (TI->doVCD) {
        vcd_parallel(&local_imageD, &local_tempD,
                     local_rows, columns,
                     TI, self, np, global_start_row, rows);
    }

    /*
     * 8) Запускаем Sobel (MPI+OpenMP) или нет:
     *     Если нужно после VCD сделать Sobel, то перед этим local_imageD
     *     уже содержит результат VCD, и мы можем выполнять Sobel.
     */
    if (TI->doSobel) {
        sobel_parallel(&local_imageD, &local_tempD,
                       local_rows, columns,
                       TI->sobelC, self, np);
    }

    /*
     * 9) Собираем собранное изображение из local_imageD без учёта ghost:
     *    Сначала готовим отдельный буфер «без ghost», затем загоняем в Gatherv.
     */
    uint8_t *full_image_out = NULL;
    double  *full_imageD_out = NULL;
    if (self == 0) {
        full_imageD_out = (double *)malloc(rows * columns * sizeof(double));
        if (full_imageD_out == NULL) {
            fprintf(stderr, "Could not allocate memory for full_imageD_out\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Сначала копируем только «реальные» строки (ghost_top..ghost_top+local_rows-1)
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

        // Собираем на rank=0
        collect_image_data(temp_send, full_imageD_out,
                           rows, columns, np, self);

        free(temp_send);
    }

    // 10) На rank=0: преобразуем full_imageD_out (double) → uint8 и записываем файл.
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

        // Опционально можно вывести общее время (если TI даёт time_loaded/time_computed)
        // printf("Computation time: %.6f\n", time_computed - time_loaded);

        free(full_image_out);
        free(full_imageD_out);
        free(full_image);
        free(full_imageD);
    }

    // 11) Локальная очистка
    free(local_imageD);
    free(local_tempD);
}
