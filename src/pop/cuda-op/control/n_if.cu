/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
   	namespace Pop{
   	    namespace Op{	
           	       		
            __global__ void If(bool * xb, float * xf, size_t idxf, size_t idxb, size_t N)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    /* if (xb[idx-1]) */
                    /*     xf[idx-1] = xf[idx-1]; */
                    /* else */
                    /*     out[i] = 0; */
	                if (!xb[(idxb-1)*N+i])
	                    xf[(idxf-1)*N+i] = 0;
                }
                return;
            }
            void GPU_If(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
            {
                If<<< DIM_GRID, DIM_BLOCK >>>(xb, xf, idxf, idxb, N);
            }
        }	
    }
}

