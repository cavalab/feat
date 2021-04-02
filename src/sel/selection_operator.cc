/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "selection_operator.h"

namespace FT{
namespace Sel{

SelectionOperator::~SelectionOperator(){}

vector<size_t> SelectionOperator::select(Population& pop, 
        const Parameters& p, const Data& d) 
{   
    THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}

vector<size_t> SelectionOperator::survive(Population& pop, 
        const Parameters& p, const Data& d)
{
    THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}
    
}

}

