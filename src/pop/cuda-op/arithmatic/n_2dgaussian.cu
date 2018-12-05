/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"
#include "../cuda_device.cuh"

namespace FT{   	
   	namespace Pop{
   	    namespace Op{	
            __global__ void Gaussian2D(float * x, size_t idx,
                                       float x1mean, float x1var,
                                       float x2mean, float x2var,
                                       float W0, float W1, size_t N)
            
            {                    
	            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	            {
	                float x1 = x[(idx-1)*N + i];
	                float x2 = x[(idx-2)*N + i];
	                
                    x[(idx-2)*N + i] = limited(exp(-1*(pow(W0*(x1-x1mean), 2)/(2*x1var) 
                                           + pow(W1*(x2 - x2mean), 2)/x1var)));
                }
                    
                return;
            }
            void GPU_Gaussian2D(float * x, size_t idx,
                                float x1mean, float x1var,
                                float x2mean, float x2var,
                                float W0, float W1, size_t N)
            {	    
                Gaussian2D<<< DIM_GRID, DIM_BLOCK >>>(x, idx, x1mean, x1var, x2mean, x2var, W0, W1, N);
            }
        }
    }
}	


