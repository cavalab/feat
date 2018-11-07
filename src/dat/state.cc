/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "state.h"
#include <iostream>

namespace FT
{
    namespace Dat{
        bool State::check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.size() >= arity['f'] &&
                        b.size() >= arity['b'] &&
                        c.size() >= arity['c']);
            else
                return (f.size() >= arity['f'] &&
                        b.size() >= arity['b'] &&
                        c.size() >= arity['c'] &&
                        z.size() >= arity['z']);
        }
        
        ///< checks if arity of node provided satisfies the node names in various string State
        bool State::check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] && 
                        bs.size() >= arity['b'] &&
                        cs.size() >= arity['c']);
            else
                return (fs.size() >= arity['f'] &&
                        bs.size() >= arity['b'] &&
                        cs.size() >= arity['c'] &&
                        zs.size() >= arity['z']);
        }
    }
}

