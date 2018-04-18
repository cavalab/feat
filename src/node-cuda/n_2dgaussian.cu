/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "error_handling.h"
#include "cuda_utils.h"
#include "cuda_utils.h"
/* #include "../node/n_2dgaussian.h" */

namespace FT{
   		
    __global__ void Gaussian2D(float * x, float x1mean, float x1var, float x2mean, float x2var, size_t idx, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            x[(idx-2)*N+i] = exp(-(pow(x[(idx-1)*N+i] - x1mean, 2) / (2 * x1var) +
                                   pow(x[(idx-2)*N+i] - x2mean, 2) / (2 * x2var))); 
        }
        return;
    }
    void GPU_Gaussian2D(float * x, float x1mean, float x1var, float x2mean, float x2var, size_t idx, size_t N)
    {
        Gaussian2D<<< DIM_GRID, DIM_BLOCK >>>(x, x1mean, x1var, x2mean, x2var, idx, N);
    }
    /// Evaluates the node and updates the stack states. 
    /* void Node2DGaussian::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, */ 
    /*         vector<ArrayXb>& stack_b) */
    /* { */
    /*     ArrayXd x2 = stack_f.back(); stack_f.pop_back(); */
    /*     ArrayXd x1 = stack_f.back(); stack_f.pop_back(); */
    /*     double x2mean = x2.mean(); */
    /*     double x2var = variance(x2); */
    /*     double x1mean = x1.mean(); */
    /*     double x1var = variance(x1); */
    /*     // evaluate on the GPU */
    /*     ArrayXd result = ArrayXd(x1.size()); */
    /*     size_t N = result.size(); */
    /*     double * dev_res; */
    /*     int numSMs; */
    /*     cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); */
    /*     // allocate device arrays */
    /*     double * dev_x1, * dev_x2 ; */ 
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(double)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(double)*N)); */
    /*     HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(double)*N)); */
    /*     //HANDLE_ERROR(cudaMalloc((void **)&x2mean, sizeof(double))); */
    /*     //HANDLE_ERROR(cudaMalloc((void **)&x2var, sizeof(double))); */
    /*     //HANDLE_ERROR(cudaMalloc((void **)&x1mean, sizeof(double))); */
    /*     //HANDLE_ERROR(cudaMalloc((void **)&x2var, sizeof(double))); */

    /*     // Copy to device */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(double)*N, cudaMemcpyHostToDevice)); */
    /*     HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(double)*N, cudaMemcpyHostToDevice)); */

    /*     Gaussian2D<<< 32*numSMs, 128 >>>(dev_x1, x1mean, x1var, dev_x2, x2mean, x2var, dev_res, N); */
       
    /*     // Copy to host */
    /*     HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(double)*N, cudaMemcpyDeviceToHost)); */
        
    /*     stack_f.push_back(limited(result)); */
    /*     // Free memory */
    /*     cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res); */
    /* } */

}	


