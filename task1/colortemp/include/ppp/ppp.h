#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>

struct TaskInput {
    /// filename: Name of the input file
    char *filename;

    /// outfilename: Name of the output file
    char *outfilename;

    /// originalColorTemp: original color temperature of input image
    int originalColorTemp;

    /// newColorTemp: desired color temperature of output image
    int newColorTemp;

    /// parallel_loading: enable parallel image loading
    bool parallel_loading;

    /// parallel_saving: enable parallel image saving
    bool parallel_saving;
};

/// \brief Converts a gray scale image with new Low/High bounds.
void compute_single(const struct TaskInput *TI);

/// \brief Converts a gray scale image with new Low/High bounds in parallel,
/// using OpenMP and MPI.
void compute_parallel(const struct TaskInput *TI);

/// \brief Converts a color temperature in Kelvin to an approximate
/// RGB color (all three components in range 0..255).
void colorTempToRGB(int tempInKelvin, double *r, double *g, double *b);

/// \brief Converts a RGB color value (all components in range 0..255)
/// to an approximate color temperature value in Kelvin.
int rgbToColorTemp(double r, double g, double b);

/// \brief Returns the number of seconds since 1970-01-01T00:00:00.
inline static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000;
}
