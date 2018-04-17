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
extern int NUM_SMS; 
extern int DIM_GRID; 
extern int DIM_BLOCK; 

static void Initialize();

static void HandleError( cudaError_t err, const char *file, int line );
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif
