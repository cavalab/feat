/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
    namespace Pop{
        namespace Op{
                
            __global__ void Relu(float * x, size_t idx, size_t N, float W0)
            {                    
	            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                    x[(idx-1)*N + i] = W0*x[(idx-1)*N + i] > 0 ? W0*x[(idx-1)*N + i] : 0.01;

                return;
            }
            void GPU_Relu(float * x, size_t idx, size_t N, float W0)
            {
                Relu<<< DIM_GRID, DIM_BLOCK >>>(x, idx, N, W0);
            }
        }	
    }
}

