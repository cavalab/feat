/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
    namespace Pop{
        namespace Op{ 
              		
            __global__ void Xor( bool * x, size_t idx, size_t N)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    x[(idx-2)*N+i] = x[(idx-1)*N+i] != x[(idx-2)*N+i];
                }
                return;
            }
            
            void GPU_Xor( bool * x, size_t idx, size_t N)
            {
                       
                Xor<<< DIM_GRID, DIM_BLOCK >>>( x, idx, N);
            }

        }	
    }
}
