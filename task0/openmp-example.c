#include <stdio.h>
#include <omp.h>

int main(void)
{
    printf("This is the main thread.\n");

#pragma omp parallel
    {
	int np   = omp_get_num_threads();
	int self = omp_get_thread_num();
	printf("This is thread #%d of %d threads in total.\n", self, np);
    }

    printf("This is again the main thread.\n");

    return 0;
}
