/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "stack.h"

namespace FT
{
    bool Stacks::check(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (f.size() >= arity['f'] && b.size() >= arity['b']);
        else
            return (f.size() >= arity['f'] && b.size() >= arity['b'] 
                    && z.size() >= arity['z']);
    }
    
    ///< checks if arity of node provided satisfies the node names in various string stacks
    bool Stacks::check_s(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
        else
            return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                    && zs.size() >= arity['z']);
    }
    
    template class Stack<ArrayXd>;
    template class Stack<ArrayXb>;
    template class Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > >;
    template class Stack<string>;
}

