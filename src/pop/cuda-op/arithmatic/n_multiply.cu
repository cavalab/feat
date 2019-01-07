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
           	      		
            __global__ void Multiply( float * x, size_t idx, size_t N, float W0, float W1)
            {     
            
                float MAX_FLT = std::numeric_limits<float>::max();
                float MIN_FLT = std::numeric_limits<float>::lowest();                  
               
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    x[(idx-2)*N+i] = (W0*x[(idx-1)*N+i] * W1*x[(idx-2)*N+i]);
                    
                    x[(idx-2)*N+i] = (isnan(x[(idx-2)*N+i])) ? 0 : x[(idx-2)*N+i];
                    x[(idx-2)*N+i] = (x[(idx-2)*N+i] < MIN_FLT) ? MIN_FLT : x[(idx-2)*N+i];
                    x[(idx-2)*N+i] = (x[(idx-2)*N+i] > MAX_FLT) ? MAX_FLT : x[(idx-2)*N+i];
                }
                return;
            }
            void GPU_Multiply( float * x, size_t idx, size_t N, float W0, float W1)
            {
                Multiply<<< DIM_GRID, DIM_BLOCK >>>(x, idx, N, W0, W1);
            }
            /// Evaluates the node and updates the stack states. 
            /* void NodeMultiply::evaluate(const MatrixXf& X, const VectorXf& y, vector<ArrayXf>& stack_f, */ 
            /*         vector<ArrayXb>& stack_b) */
            /* { */
            /*     ArrayXf x2 = stack_f.back(); stack_f.pop_back(); */
            /*     ArrayXf x1 = stack_f.back(); stack_f.pop_back(); */
            /*     // evaluate on the GPU */
            /*     ArrayXf result = ArrayXf(x1.size()); */
            /*     size_t N = result.size(); */
            /*     float * dev_res; */
            /*     int numSMs; */
            /*     cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); */
            /*     // allocate device arrays */
            /*     float * dev_x1, * dev_x2 ; */ 
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(float)*N)); */
            /*     // Copy to device */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */

            /*     Multiply<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N); */
               
            /*     // Copy to host */
            /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(float)*N, cudaMemcpyDeviceToHost)); */
                
            /*     stack_f.push_back(limited(result)); */
            /*     // Free memory */
            /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
            /* } */

        }	
    }
}

