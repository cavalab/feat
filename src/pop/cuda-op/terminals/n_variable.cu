/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
    namespace Pop{
        namespace Op{    

            void GPU_Variable(float * dev_x, float * host_x, size_t idx, size_t N)
            {
	            //printf("GPU variable called for float with values %u %u %d %d\n", dev_x, host_x, idx, N);
                HANDLE_ERROR(cudaMemcpy(dev_x+idx*N, host_x, sizeof(float)*N, cudaMemcpyHostToDevice));
            }
            
            void GPU_Variable(int * dev_x, int * host_x, size_t idx, size_t N)
            {
	            //printf("GPU variable called for int with values %u %u %d %d\n", dev_x, host_x, idx, N);
                HANDLE_ERROR(cudaMemcpy(((int*) dev_x+idx*N), host_x, sizeof(int)*N, cudaMemcpyHostToDevice));
            }

            void GPU_Variable(bool * dev_x, bool * host_x, size_t idx, size_t N)
            {
                //printf("GPU variable called for boolean\n");
                HANDLE_ERROR(cudaMemcpy(dev_x+idx*N, host_x, sizeof(bool)*N, cudaMemcpyHostToDevice));
            }
        }
    }
}
