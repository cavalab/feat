/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "error.h"

//#include "node/node.h"
//external includes

namespace FT{

    namespace Util{
            /// prints error and throws an exception
            void HandleErrorThrow(string err, const char *file, int line )
            {
                std::cerr << err << " in "<< file << " at line "<<line<<"\n";
                throw;
            }
            
            ///prints error to stderr and returns
            void HandleErrorNoThrow(string err, const char *file, int line )
            {
                std::cerr << err << " in "<< file << " at line "<<line<<"\n";
        }
    }
}

