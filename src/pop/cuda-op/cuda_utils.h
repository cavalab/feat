/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

extern int NUM_SMS; 
extern int DIM_GRID; 
extern int DIM_BLOCK; 


namespace FT{
    
    namespace Pop{
        namespace Op{

            void choose_gpu();

            void initialize_cuda();
            
        }
    }
}

#endif
