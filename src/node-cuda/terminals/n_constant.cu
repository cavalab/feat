/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
   		
    __global__ void ConstantF(float * x, float value, size_t idx, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            x[(idx)*N + i] = value;
        }
        return;
    }

    __global__ void ConstantB(bool * x, bool value, size_t idx, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            x[(idx)*N + i] = value;
        }
        return;
    }

    void GPU_Constant(float * dev_x, float value, size_t idx, size_t N)
    {
        ConstantF<<< DIM_GRID, DIM_BLOCK >>>(dev_x, value, idx, N);
    }

    void GPU_Constant(bool * dev_x, bool value, size_t idx, size_t N)
    {
        ConstantB<<< DIM_GRID, DIM_BLOCK >>>(dev_x, value, idx, N);
    }
    /* void GPU_Constant(bool * dev_x, bool * host_x, size_t idx, size_t N) */
    /* { */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x+idx*N, host_x, sizeof(bool)*N, cudaMemcpyHostToDevice)); */
    /* } */
}
