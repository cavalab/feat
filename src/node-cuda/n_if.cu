/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "error_handling.h"
#include "cuda_utils.h"
/* #include "../node/n_if.h" */

namespace FT{
   		
    __global__ void If(bool * xb, float * xf, size_t idxf, size_t idxb, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            /* if (xb[idx-1]) */
            /*     xf[idx-1] = xf[idx-1]; */
            /* else */
            /*     out[i] = 0; */ 
            if (~xb[(idxb)*N+i])
                xf[(idxf-1)*N+i] = 0;
        }
        return;
    }
    void GPU_If(bool * xb, float * xf, size_t idxf, size_t idxb, size_t N)
    {
        If<<< DIM_GRID, DIM_BLOCK >>>(xb, xf, idxf, idxb, N);
    }
    /// Evaluates the node and updates the stack states. 
    /* void NodeIf::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, */ 
    /*         vector<ArrayXb>& stack_b) */
    /* { */
    /*     ArrayXd x2 = stack_f.back(); stack_f.pop_back(); */
    /*     ArrayXb x1 = stack_b.back(); stack_b.pop_back(); */
    /*     // evaluate on the GPU */
    /*     ArrayXd result = ArrayXd(x1.size()); */
    /*     size_t N = result.size(); */
    /*     double * dev_res; */
    /*     int numSMs; */
    /*     cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); */
    /*     // allocate device arrays */
    /*     bool * dev_x1; */
    /*     double * dev_x2 ; */ 
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(bool)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(double)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(double)*N)); */
    /*     // Copy to device */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(bool)*N, cudaMemcpyHostToDevice)); */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(double)*N, cudaMemcpyHostToDevice)); */

    /*     If<<< 32*numSMs, 128 >>>(dev_x1, dev_x2, dev_res, N); */
       
    /*     // Copy to host */
    /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(bool)*N, cudaMemcpyDeviceToHost)); */
        
    /*     stack_f.push_back(result); */
    /*     // Free memory */
    /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
    /* } */

}	


