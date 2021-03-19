/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "error.h"
#include <stdexcept>

//#include "node/node.h"
//external includes

namespace FT{

namespace Util{
/// prints error and throws an exception
void ThrowRuntimeError(string err, const char *file, int line)
{
    string str_file(file);
    err = "ERROR: " + str_file + ", line " + std::to_string(line) + ": " + err;
    throw std::runtime_error(err);
}
void ThrowInvalidArgument(string err, const char *file, int line)
{
    string str_file(file);
    err = "ERROR: " + str_file + ", line " + std::to_string(line) + ": " + err;
    throw std::invalid_argument(err);
}
void ThrowLengthError(string err, const char *file, int line)
{
    string str_file(file);
    err = "ERROR: " + str_file + ", line " + std::to_string(line) + ": " + err;
    throw std::length_error(err);
}           
///prints error to stderr and returns
void Warn(string err, const char *file, int line )
{
    string str_file(file);
    err = "WARNING: " + str_file + ", line " + std::to_string(line) + ": " + err;
    std::cout << err << "\n";
}

/// handle signals (ctr-c etc.)
void my_handler(int s)
{
    WARN("Caught signal "+ to_string(s) +", exiting");
    exit(1); 
}

}
}

