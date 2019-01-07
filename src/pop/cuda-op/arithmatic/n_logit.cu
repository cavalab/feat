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
           	      		
            __global__ void Logit( float * x, size_t idx, size_t N, float W0)
            {                    
            
                float MAX_FLT = std::numeric_limits<float>::max();
                float MIN_FLT = std::numeric_limits<float>::lowest();                  

                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    float ex = exp(-W0*x[(idx-1)*N+i]);
                    
                    ex = (isnan(ex)) ? 0 : ex;
                    ex = (ex < MIN_FLT) ? MIN_FLT : ex;
                    ex = (ex > MAX_FLT) ? MAX_FLT : ex;
                    
                    x[(idx-1)*N+i] = (1 / (1 + ex));
                }
                return;
            }
            void GPU_Logit( float * x, size_t idx, size_t N, float W0)
            {
                Logit<<< DIM_GRID, DIM_BLOCK >>>(x, idx, N, W0);
            }
            /// Evaluates the node and updates the stack states. 
            /* void NodeLogit::evaluate(const MatrixXf& X, const VectorXf& y, vector<ArrayXf>& stack_f, */ 
            /*         vector<ArrayXb>& stack_b) */
            /* { */
            /*     ArrayXf x1 = stack_f.back(); stack_f.pop_back(); */
            /*     // evaluate on the GPU */
            /*     ArrayXf result = ArrayXf(x1.size()); */
            /*     size_t N = result.size(); */
            /*     float * dev_res; */
            /*     int numSMs; */
            /*     cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); */
            /*     // allocate device arrays */
            /*     float * dev_x1 ; */ 
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(float)*N)); */
            /*     // Copy to device */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */

            /*     Logit<<< 32*numSMs, 128 >>>(dev_x1, dev_res, N); */
               
            /*     // Copy to host */
            /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(float)*N, cudaMemcpyDeviceToHost)); */
                
            /*     stack_f.push_back(limited(result)); */
            /*     // Free memory */
            /*     cudaFree(dev_x1); cudaFree(dev_res); */
            /* } */

        }	
    }
}
