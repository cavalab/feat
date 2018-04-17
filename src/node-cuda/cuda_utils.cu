/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"

static void FT::Initialize()
{
    cudaDeviceGetAttribute(&NUM_SMS, cudaDevAttrMultiProcessorCount, 0); 
    DIM_GRID = 32*NUM_SMS;
    DIM_BLOCK = 128; 

}



