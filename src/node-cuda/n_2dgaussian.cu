/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
#include "../node/n_2dgaussian.h"

namespace FT{
   		
    __global__ void 2DGaussian(double * x1, double x1_mean, double x1_var,
                               double * x2, double x2_mean, double x2_var, double * out, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            out[i] = exp(-(pow(x1[i] - x1_mean, 2) / (2 * x1_var) +
                           pow(x2[i] - x2_mean, 2) / (2 * x2_var))); 
        }
        return;
    }
    /// Evaluates the node and updates the stack states. 
    void Node2DGaussian::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
            vector<ArrayXb>& stack_b)
    {
        ArrayXd x2 = stack_f.back(); stack_f.pop_back();
        ArrayXd x1 = stack_f.back(); stack_f.pop_back();
        double x2_mean = x2.mean();
        double x2_var = variance(x2);
        double x1_mean = x1.mean();
        double x1_var = variance(x1);
        // evaluate on the GPU
        ArrayXd result = ArrayXd(x1.size());
        size_t N = result.size();
        double * dev_res;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        // allocate device arrays
        double * dev_x1, * dev_x2 ; 
        HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(double)*N));
        HANDLE_ERROR(cudaMalloc((void **)& dev_x2, sizeof(double)*N));
        HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(double)*N));
        //HANDLE_ERROR(cudaMalloc((void **)&x2_mean, sizeof(double)));
        //HANDLE_ERROR(cudaMalloc((void **)&x2_var, sizeof(double)));
        //HANDLE_ERROR(cudaMalloc((void **)&x1_mean, sizeof(double)));
        //HANDLE_ERROR(cudaMalloc((void **)&x2_var, sizeof(double)));

        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(double)*N, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_x2, x2.data(), sizeof(double)*N, cudaMemcpyHostToDevice));

        2DGaussian<<< 32*numSMs, 128 >>>(dev_x1, x1_mean, x1_var, dev_x2, x2_mean, x2_var, dev_res, N);
       
        // Copy to host
        HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(double)*N, cudaMemcpyDeviceToHost));
        
        stack_f.push_back(limited(result));
        // Free memory
        cudaFree(dev_x1); cudaFree(dev_x2); cudaFree(dev_res);
    }

}	


