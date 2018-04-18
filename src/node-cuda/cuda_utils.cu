/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"

static void FT::Initialize()
{
    cudaDeviceGetAttribute(&NUM_SMS, cudaDevAttrMultiProcessorCount, 0); 
    DIM_GRID = 32*NUM_SMS;
    DIM_BLOCK = 128; 

}

static void FT::ChooseGPU()
{
        int n_gpus; 
        cudaGetDeviceCount(&n_gpus);
        int device = omp_get_max_threads() % n_gpus ; 
        cudaSetDevice(device); 

}


