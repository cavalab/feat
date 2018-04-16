/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
/* #include "../node/n_leq.h" */

namespace FT{
   		
    __global__ void LEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            xb[idxb*N+i] = xf[(idxf-1)*N+i] <= xf[(idxf-2)*N+i];
        }
        return;
    }
    void GPU_LEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
    {
        GPU_LEQ<<< DIM_GRID, DIM_BLOCK, omp_get_thread_num() >>>(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
    }
    /// Evaluates the node and updates the stack states. 
    /* void NodeLEQ::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, */ 
    /*         vector<ArrayXb>& stack_b) */
    /* { */
    /*     ArrayXd x2 = stack_f.back(); stack_f.pop_back(); */
    /*     ArrayXd x1 = stack_f.back(); stack_f.pop_back(); */
    /*     // evaluate on the GPU */
    /*     ArrayXb result = ArrayXb(x1.size()); */
    /*     size_t N = result.size(); */
    /*      * dev_x2 ; */ 
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(double)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(double)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(bool)*N)); */
    /*     // Copy to device */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(double)*N, cudaMemcpyHostToDevice)); */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(double)*N, cudaMemcpyHostToDevice)); */

    /*     LEQ<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N); */
       
    /*     // Copy to host */
    /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(bool)*N, cudaMemcpyDeviceToHost)); */
        
    /*     stack_b.push_back(result); */
    /*     // Free memory */
    /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
    /* } */

}	


