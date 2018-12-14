/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
   	namespace Pop{
   	    namespace Op{	
           		
            __global__ void Divide( float * x, size_t idx, size_t N, float W0, float W1)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    if (abs(x[(idx-2)*N+i]) > 0.00000001)
                        x[(idx-2)*N+i] = ((W0*x[(idx-1)*N+i]) / (W1*x[(idx-2)*N+i]));
                    else
                        x[(idx-2)*N+i] = 1.0 ;  
                    //printf("Result is %f\n", x[(idx-2)*N+i]);
                }
                return;
            }
            void GPU_Divide( float * x, size_t idx, size_t N, float W0, float W1)
            {
                Divide<<< DIM_GRID, DIM_BLOCK >>>(x, idx, N, W0, W1);
            }
            /// Evaluates the node and updates the stack states. 
            /* void NodeDivide::evaluate(const MatrixXf& X, const VectorXf& y, vector<ArrayXf>& stack_f, */ 
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

            /*     Divide<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N); */
               
            /*     // Copy to host */
            /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(float)*N, cudaMemcpyDeviceToHost)); */
                
            /*     stack_f.push_back(limited(result)); */
            /*     // Free memory */
            /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
            /* } */

        }	
    }
}
