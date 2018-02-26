/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
#include "n_cos.h"

namespace FT{
   		
    __global__ void Cos(double * x1, double * out, size_t N)
    {                    
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            out[i] = cos ( x1[i] );
        }
        return;
    }
    /// Evaluates the node and updates the stack states. 
    void NodeCos::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
            vector<ArrayXb>& stack_b)
    {
        ArrayXd x1 = stack_f.back(); stack_f.pop_back();
        // evaluate on the GPU
        ArrayXd result = ArrayXd(x1.size());
        size_t N = result.size();
        double * dev_res;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        // allocate device arrays
        double * dev_x1; 
        HANDLE_ERROR(cudaMalloc((void **)& dev_x1, sizeof(double)*N));
        HANDLE_ERROR(cudaMalloc((void **)&dev_res, sizeof(double)*N));
        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_x1, x1.data(), sizeof(double)*N, cudaMemcpyHostToDevice));

        Cos<<< 32*numSMs, 128 >>>(dev_x1, dev_res, N);
       
        // Copy to host
        HANDLE_ERROR(cudaMemcpy(result.data(), dev_res, sizeof(double)*N, cudaMemcpyDeviceToHost));
        
        stack_f.push_back(limited(result));
        // Free memory
        cudaFree(dev_x1); cudaFree(dev_res);
    }

}	
#endif

