/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>


// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    exit(0);
  }

  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  // Pyramid parameters.
  const int borderCols = (pyramid_height) * HALO;

  /* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
     pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

  int size = rows * cols; // the size (global work size) is a multiple of lws

  // running the opencl application shows block_size=4000 (cpu) and block_size=250 (gpu)
#ifdef USE_GPU
  int block_size = 250;
#else
  int block_size = 4000;
#endif
  int* outputBuffer = (int*)calloc(16384, sizeof(int));
  int theHalo = HALO;

  auto start = std::chrono::steady_clock::now();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_gpuWall = sycl::malloc_device<int>(size-cols, q);
  q.memcpy(d_gpuWall, data+cols, sizeof(int)*(size-cols));

  int *d_gpuSrc = sycl::malloc_device<int>(cols, q);
  q.memcpy(d_gpuSrc, data, sizeof(int)*cols);

  int *d_gpuResult = sycl::malloc_device<int>(cols, q);
  int *d_outputBuffer = sycl::malloc_device<int>(16384, q);

  sycl::range<1> gws(size);
  sycl::range<1> lws(block_size);

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    // Calculate this for the kernel argument...
    int iteration = MIN(pyramid_height, rows-t-1);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor <int, 1> prev (lws, cgh);
      sycl::local_accessor <int, 1> result (lws, cgh);
      // Set the kernel arguments.
      cgh.parallel_for<class dynproc_kernel>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          #include "kernel.sycl"
      });
    });

    int* temp = d_gpuResult;
    d_gpuResult = d_gpuSrc;
    d_gpuSrc = temp;
  } // for

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Total kernel execution time: %lf (s)\n", ktime * 1e-9);

  q.memcpy(result, d_gpuSrc, sizeof(int)*cols);
  q.memcpy(outputBuffer, d_outputBuffer, sizeof(int)*16348);
  q.wait();

  sycl::free(d_gpuResult, q);
  sycl::free(d_gpuSrc, q);
  sycl::free(d_gpuWall, q);
  sycl::free(d_outputBuffer, q);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time = %lf (s)\n", time * 1e-9);

  // add a null terminator at the end of the string.
  outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", result[i]);
  printf("\n");
#endif

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  free(outputBuffer);

  return EXIT_SUCCESS;
}
