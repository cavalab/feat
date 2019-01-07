/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
   	namespace Pop{
   	    namespace Op{	
           	       		
            __global__ void Split(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N, float threshold)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    xb[(idxb)*N+i] = (xf[(idxf-1)*N+i] < threshold);
                }
                return;
            }
            
            __global__ void Split(int * xi, bool * xb, size_t idxi, size_t idxb, size_t N, float threshold)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    xb[(idxb)*N+i] = (((float)xi[(idxi-1)*N+i]) == threshold);
                }
                return;
            }
            
            void GPU_Split(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N, float threshold)
            {
                Split<<< DIM_GRID, DIM_BLOCK >>>(xf, xb, idxf, idxb, N, threshold);
            }
            
            void GPU_Split(int * xi, bool * xb, size_t idxi, size_t idxb, size_t N, float threshold)
            {
                Split<<< DIM_GRID, DIM_BLOCK >>>(xi, xb, idxi, idxb, N, threshold);
            }
        }	
    }
}

