#include <math.h>

void colorTempToRGB(int tempInKelvin, double *r, double *g, double *b) {
    // Conversion formulas from
    // https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html

    double temp = tempInKelvin / 100.0;
    if (temp <= 66) {
        *r = 255.0;
        *g = fmin(255, 99.4708025861 * log(temp) - 161.1195681661);
    } else {
        *r = fmin(255, 329.698727446 * pow(temp - 60, -0.1332047592));
        *g = fmin(255, 288.1221695283 * pow(temp - 60, -0.0755148492));
    }
    if (temp >= 66) {
        *b = 255.0;
    } else if (temp <= 19) {
        *b = 0.0;
    } else {
        *b = fmin(255, 138.5177312231 * log(temp - 10) - 305.0447927307);
    }
}

int rgbToColorTemp(double r, double g, double b) {
    // McCamy's approximation of the correlated color temperature, cf.
    // doi:10.1002/col.5080170211

    double n = (0.23881 * r + 0.25499 * g - 0.58291 * b) / (0.11109 * r - 0.85406 * g + 0.52289 * b);
    return rint(((449 * n + 3525) * n + 6823.3) * n + 5520.33);
}
