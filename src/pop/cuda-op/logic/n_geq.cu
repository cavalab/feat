/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "../error_handling.h"
#include "../cuda_utils.h"

namespace FT{
    namespace Pop{
        namespace Op{
                        
            __global__ void GEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
            {                    
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
                {
                    xb[idxb*N+i] = xf[(idxf-1)*N+i] >= xf[(idxf-2)*N+i];
                }
                return;
            }
            void GPU_GEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
            {
                GEQ<<< DIM_GRID, DIM_BLOCK >>>(xf, xb, idxf, idxb, N);
            }
            /// Evaluates the node and updates the stack states. 
            /* void NodeGEQ::evaluate(const MatrixXf& X, const VectorXf& y, vector<ArrayXf>& stack_f, */ 
            /*         vector<ArrayXb>& stack_b) */
            /* { */
            /*     ArrayXf x2 = stack_f.back(); stack_f.pop_back(); */
            /*     ArrayXf x1 = stack_f.back(); stack_f.pop_back(); */
            /*     // evaluate on the GPU */
            /*     ArrayXb result = ArrayXb(x1.size()); */
            /*     size_t N = result.size(); */
            /*     bool * dev_res; */
            /*     int numSMs; */
            /*     cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); */
            /*     // allocate device arrays */
            /*     float * dev_x1, * dev_x2 ; */ 
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(float)*N)); */
            /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(bool)*N)); */
            /*     // Copy to device */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */
            /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(float)*N, cudaMemcpyHostToDevice)); */

            /*     GEQ<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N); */
               
            /*     // Copy to host */
            /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(bool)*N, cudaMemcpyDeviceToHost)); */
                
            /*     stack_b.push_back(result); */
            /*     // Free memory */
            /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
            /* } */

        }	
    }
}


