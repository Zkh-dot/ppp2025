#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ppp_pnm/ppp_pnm.h"

/*
 * Load a PGM (Portable Graymap) image and invert
 * the gray values of every pixel.
 * The program is called with 2 arguments:
 *      Input-image  Output-image
 */
int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image;
    int x, y;

    if (argc != 3) {
	fprintf(stderr, "USAGE: %s IN OUT\n", argv[0]);
	return 1;
    }

    /*
     * Load the image (name in argv[1]),
     * store the kind (PBM, PGM, PPM) in 'kind',
     * the number of rows and columns in 'rows' and 'columns',
     * the maximal gray value of the image format (NOT the
     * maximal gray value used in the image) in 'maxcolor' and return
     * the image row-wise with one int per pixel.
     */
    image = ppp_pnm_read(argv[1], &kind, &rows, &columns, &maxcolor);
    
    if (image != NULL) {
	if (kind == PNM_KIND_PGM) {
	    for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
		    image[y*columns+x] = maxcolor - image[y*columns+x];
		}
	    }
	    
	    /*
	     * Save the image, parameters are analogous to
	     * ppp_pnm_read (but kind, rows, columns, maxcolor are
	     * not passed as pointers for ppp_pnm_write). The
	     * last parameter is a pointer to the image to be saved.
	     */
	    if (ppp_pnm_write(argv[2], kind, rows, columns, maxcolor, image) != 0)
		fprintf(stderr, "write error\n");
	} else
	    fprintf(stderr, "not a PGM image\n");

	free(image);
    } else
	fprintf(stderr, "could not load image\n");

    return 0;
}
