/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <iostream>
static void HandleError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif

