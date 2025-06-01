#include <math.h>
#include <stdlib.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

void compute_single(const struct TaskInput *TI) {
    enum pnm_kind kind;
    int columns;
    int rows;
    int maxval;
    uint8_t *image;

    // Read the image parameters and the pixel values from the input file.
    image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxval);
    if (image == NULL) {
        fprintf(stderr, "Could not load image from file '%s'.\n", TI->filename);
        return;
    } else if (kind != PNM_KIND_PPM) {
        fprintf(stderr, "Image is not a \"portable pixmap\".\n");
        return;
    }

    // Sum up the red, green and blue values of all pixels in
    // sums[0], sums[1], sums[2], respectively.
    long sums[3] = {0, 0, 0};
    for (int i = 0; i < rows * columns; ++i) {
        for (int j = 0; j <= 2; ++j) {
            sums[j] += image[3 * i + j];
        }
    }

    // Compute the average red, green and blue value of the image.
    int pixels = rows * columns;
    double avgRed = (double)sums[0] / pixels;
    double avgGreen = (double)sums[1] / pixels;
    double avgBlue = (double)sums[2] / pixels;

    // Estimate the color temperature of the image.
    double estimatedTemp = rgbToColorTemp(avgRed, avgGreen, avgBlue);
    printf("Estimated color temperature: %g K\n", estimatedTemp);

    // To convert from the given original color temperature to a new
    // color temperature below, we compute the RGB representation of both
    // color temperatures.
    double from[3], to[3];
    colorTempToRGB(TI->originalColorTemp, &from[0], &from[1], &from[2]);
    colorTempToRGB(TI->newColorTemp, &to[0], &to[1], &to[2]);
    printf("Adjusting color temperature from %d K to %d K.\n", TI->originalColorTemp, TI->newColorTemp);
    printf("Original color temperature %d K corresponds to RGB color (%g, %g, %g).\n", TI->originalColorTemp, from[0],
           from[1], from[2]);
    printf("New color temperature %d K corresponds to RGB color (%g, %g, %g).\n", TI->newColorTemp, to[0], to[1],
           to[2]);

    // Each component (red, green, blue) of every pixel is multiplied by the
    // corresponding 'to' value and divided by the 'from' value. Computed values
    // can be greater than the maximal allowed value (maxval); therefore, we have
    // to use fmin(maxval, ...) to ensure we do not overflow pixel values.
    for (int i = 0; i < rows * columns; ++i) {
        for (int j = 0; j <= 2; ++j) {
            image[3 * i + j] = rint(fmin(maxval, image[3 * i + j] * to[j] / from[j]));
        }
    }

    // Write converted image data to the output file.
    if (ppp_pnm_write(TI->outfilename, kind, rows, columns, maxval, image) != 0) {
        fprintf(stderr, "Could not write output image to %s\n", TI->outfilename);
    }

    free(image);
}
