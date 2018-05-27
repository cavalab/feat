/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "selection/selection_operator.h"
#include "selection/lexicase.h"
#include "selection/nsga2.h"
#include "selection/offspring.h"
#include "selection/random.h"
#include "selection/simulated_annealing.h"

namespace FT{
    struct Parameters; // forward declaration of Parameters       
    ////////////////////////////////////////////////////////////////////////////////// Declarations
	
    
    /*!
     * @class Selection
     * @brief interfaces with selection operators. 
     */
    struct Selection
    {
        shared_ptr<SelectionOperator> pselector; 
        
        Selection(string type="lexicase", bool survival=false);
        
        ~Selection();
        
        /// return type of selectionoperator
        string get_type();
        
        /// perform selection 
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params);
        
        /// perform survival
        vector<size_t> survive(Population& pop, const MatrixXd& F,  const Parameters& params);
    };
    
}
#endif
