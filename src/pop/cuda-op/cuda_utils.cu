/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"

int NUM_SMS = 32; 
int DIM_GRID = 1024; 
int DIM_BLOCK = 128; 


namespace FT{
    namespace Pop{
        namespace Op{
        
            void initialize_cuda()
            {
                cudaDeviceGetAttribute(&NUM_SMS, cudaDevAttrMultiProcessorCount, 0); 
                DIM_GRID = 32*NUM_SMS;
                DIM_BLOCK = 128; 
            }

            void choose_gpu()
            {
	            int n_gpus; 
	            cudaGetDeviceCount(&n_gpus);
	            int device = omp_get_thread_num() % n_gpus ; 
	            cudaSetDevice(device);
            }           
            
        }
    }
}


