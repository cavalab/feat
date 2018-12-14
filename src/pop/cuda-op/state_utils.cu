/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "state_utils.h"
#include "error_handling.h"
// stack utils

namespace FT{
    namespace Pop{
        namespace Op{

            void dev_allocate(float *& f, size_t Sizef,
                              int *& c, size_t Sizec,
                              bool *& b, size_t Sizeb)
            {
                HANDLE_ERROR(cudaMalloc((void **)& f, sizeof(float)*Sizef));
                HANDLE_ERROR(cudaMalloc((void **)& c, sizeof(float)*Sizec));
                HANDLE_ERROR(cudaMalloc((void **)& b, sizeof(bool)*Sizeb));
	            //HANDLE_ERROR(cudaDeviceSynchronize());
                //std::cout << "allocated " << sizeof(float)*Sizef << " bytes at loc " << f << " for stack.f\n";
                //std::cout << "allocated " << sizeof(bool)*Sizeb << " bytes at loc " << b << " for stack.b\n";
            }
            
            void copy_from_device(float * dev_f, float * host_f, size_t Sizef)
            {
                HANDLE_ERROR(cudaMemcpy(host_f, dev_f, sizeof(float)*Sizef, cudaMemcpyDeviceToHost));
            }
            
            void copy_from_device(int * dev_c, int * host_c, size_t Sizec)
            {
                HANDLE_ERROR(cudaMemcpy(host_c, dev_c, sizeof(int)*Sizec, cudaMemcpyDeviceToHost));
            }
            
            void copy_from_device(bool * dev_b, bool * host_b, size_t Sizeb)
            {
                HANDLE_ERROR(cudaMemcpy(host_b, dev_b, sizeof(bool)*Sizeb,  cudaMemcpyDeviceToHost));
            }

            void copy_from_device(float * dev_f, float * host_f, size_t Sizef,
                                  int * dev_c, int * host_c, size_t Sizec,
                                  bool * dev_b, bool * host_b, size_t Sizeb)
            {
                //std::cout << "dev_f: " << dev_f << "\nhost_f: " << host_f << "\nSizef: " << Sizef << "\nSizeb: " << Sizeb <<"\n";
                
	            copy_from_device(dev_f, host_f, Sizef);
	            copy_from_device(dev_c, host_c, Sizec);
	            copy_from_device(dev_b, host_b, Sizeb);	
	            //HANDLE_ERROR(cudaDeviceSynchronize());
            }

            void free_device(float * dev_f, int * dev_c, bool * dev_b)
            {
                // Free memory
                cudaFree(dev_f); 
                cudaFree(dev_c);
                cudaFree(dev_b);         
	            //HANDLE_ERROR(cudaDeviceSynchronize());
            }
        }
    }
}
