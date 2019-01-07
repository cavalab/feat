/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
   	namespace Pop{
   	    namespace Op{	
           	    		
            __global__ void IfThenElse(bool * b, float * x, size_t idxb, size_t idxf, size_t N)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    //printf("From IfThenElse %d, %f, %f\n", b[(idxb-1)*N+i], x[(idxf-1)*N+i], x[(idxf-2)*N+i]);
                    if (b[(idxb-1)*N+i])
                        x[(idxf-2)*N+i] = x[(idxf-1)*N+i];
                    //printf("After IfThenElse %f\n", x[(idxf-2)*N+i]);
                        
                }
                return;
            }
            void GPU_IfThenElse(float * x, bool *b ,size_t idxf, size_t idxb, size_t N)
            {
                IfThenElse<<< DIM_GRID, DIM_BLOCK >>>(b, x, idxb, idxf, N);
            }
            /// Evaluates the node and updates the stack states. 
            /* void NodeIfThenElse::evaluate(const MatrixXf& X, const VectorXf& y, vector<ArrayXf>& stack_f, */ 
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
            /*     bool * dev_b1; */

            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_b1, sizeof(bool)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(float)*N)); */
            /*     // Copy to device */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */

            /*     IfThenElse<<< 32*numSMs, 128 >>>(dev_b1, dev_x1, dev_x2, dev_res, N); */
               
            /*     // Copy to host */
            /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(float)*N, cudaMemcpyDeviceToHost)); */
                
            /*     stack_f.push_back(result); */
            /*     // Free memory */
            /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
            /* } */

        }	
    }
}

