#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"

int main( int argc, char** argv) 
{
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", BLOCK_SIZE, BLOCK_SIZE2);

	setup(argc, argv);

	return 0;
}

