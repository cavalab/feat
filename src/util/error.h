/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef ERROR_H
#define ERROR_H

#include <string>
#include <iostream>
#include <stdexcept>
#include <signal.h>

using namespace std;

//#include "node/node.h"
//external includes

namespace FT
{
namespace Util{
/// prints error and throws an exception
void ThrowRuntimeError(string err, const char *file, int line);
void ThrowInvalidArgument(string err, const char *file, int line);
void ThrowLengthError(string err, const char *file, int line);

///prints warning to stdout and returns
void Warn(string err, const char *file, int line);

#define THROW_RUNTIME_ERROR( err ) (FT::Util::ThrowRuntimeError( err, __FILE__, __LINE__ ))
#define THROW_INVALID_ARGUMENT( err ) (FT::Util::ThrowInvalidArgument( err, __FILE__, __LINE__ ))
#define THROW_LENGTH_ERROR( err ) (FT::Util::ThrowLengthError( err, __FILE__, __LINE__ ))
#define WARN( err ) (FT::Util::Warn( err, __FILE__, __LINE__ ))

/// handle signals (ctr-c etc.)
void my_handler(int s);

}
}

#endif
