/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"

namespace FT{
    

    void GPU_Constant(float * dev_x, float * host_x, size_t idx, size_t N)
    {
        HANDLE_ERROR(CudaMemcpy(dev_x+idx*N, host_x, sizeof(float)*N, cudaMemcpyDeviceToHost));
    }

    void GPU_Constant(bool * dev_x, bool * host_x, size_t idx, size_t N)
    {
        HANDLE_ERROR(CudaMemcpy(dev_x+idx*N, host_x, sizeof(bool)*N, cudaMemcpyDeviceToHost));
    }

}
