/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"
#include <limits>

namespace FT{
    namespace Pop{
        namespace Op{
                        
           		
            __global__ void ConstantF(float * x, float value, size_t idx, size_t N)
            {       
            
                float MAX_FLT = std::numeric_limits<float>::max();
                float MIN_FLT = std::numeric_limits<float>::lowest();                  
             
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    x[(idx)*N + i] = value;
                    
                    x[(idx)*N+i] = (isnan(x[(idx)*N+i])) ? 0 : x[(idx)*N+i];
                    x[(idx)*N+i] = (x[(idx)*N+i] < MIN_FLT) ? MIN_FLT : x[(idx)*N+i];
                    x[(idx)*N+i] = (x[(idx)*N+i] > MAX_FLT) ? MAX_FLT : x[(idx)*N+i];
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
    }
}
