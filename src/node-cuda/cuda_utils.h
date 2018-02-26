/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

static void HandleError( cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

