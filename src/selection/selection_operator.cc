/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "selection_operator.h"

namespace FT{

    SelectionOperator::~SelectionOperator(){}
    
    vector<size_t> SelectionOperator::select(Population& pop, const MatrixXd& F, const Parameters& p) 
    {   
        HANDLE_ERROR_THROW("Undefined select() operation");
    }
    
    vector<size_t> SelectionOperator::survive(Population& pop, const MatrixXd& F, const Parameters& p)
    {
        HANDLE_ERROR_THROW("Undefined select() operation");
    }
}

