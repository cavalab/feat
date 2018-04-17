/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
#include "stack_utils.h"
// stack utils 
namespace FT{
    void dev_allocate(float * f, bool * b, size_t Sizef, size_t Sizeb)
    {
        HANDLE_ERROR(cudaMalloc((void **)& f, sizeof(float)*Sizef));
        HANDLE_ERROR(cudaMalloc((void **)& b, sizeof(bool)*Sizeb));
    }

    void copy_from_device(float * dev_f, float * host_f, bool * dev_b, bool * host_b, size_t Sizef, 
                          size_t Sizeb)
    {
        HANDLE_ERROR(cudaMemcpy(dev_f, host_f, sizeof(float)*Sizef, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(dev_b, host_b, sizeof(bool)*Sizeb,  cudaMemcpyDeviceToHost));
    }

    void free_device(float * dev_f, bool * dev_b)
    {
        // Free memory
        cudaFree(dev_f); 
        cudaFree(dev_b);         
    }
}
