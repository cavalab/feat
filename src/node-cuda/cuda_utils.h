/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif
#include "kernels.h"

extern int NUM_SMS; 
extern int DIM_GRID; 
extern int DIM_BLOCK; 


namespace FT{
static void Initialize();

static void HandleError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
}
#endif
