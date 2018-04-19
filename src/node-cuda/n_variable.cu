/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "error_handling.h"
#include "cuda_utils.h"

namespace FT{
    

    void GPU_Variable(float * dev_x, float * host_x, size_t idx, size_t N)
    {
        std::cout << "dev_x: " << dev_x << "\nhost_x: " << host_x << "\nidx: " << idx << "\nN: " << N << "\n";
        std::cout << "dev_x+idx*N: " << dev_x + idx*N << "\n";
        HANDLE_ERROR(cudaMemcpy(dev_x+idx*N, host_x, sizeof(float)*N, cudaMemcpyHostToDevice));
    }

    void GPU_Variable(bool * dev_x, bool * host_x, size_t idx, size_t N)
    {
        HANDLE_ERROR(cudaMemcpy(dev_x+idx*N, host_x, sizeof(bool)*N, cudaMemcpyHostToDevice));
    }

}
