/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "selection/selection_operator.h"
#include "selection/lexicase.h"
#include "selection/pareto.h"
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
        
        Selection(string type="lexicase", bool survival=false)
        {
            /*!
             * set type of selection operator.
             */

            if (!type.compare("lexicase"))
                pselector = std::make_shared<Lexicase>(survival); 
            else if (!type.compare("pareto"))
                pselector = std::make_shared<Pareto>(survival);
            else
                std::cerr << "Undefined Selection Operator " + type + "\n";
                
        };

        ~Selection(){}
        
        /// return type of selectionoperator
        string get_type(){ return pselector->name; }
        
        /// perform selection 
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params)
        {       
            return pselector->select(pop, F, params);
        }
        /// perform survival
        vector<size_t> survive(Population& pop, const MatrixXd& F,  const Parameters& params)
        {       
            return pselector->survive(pop, F, params);
        }
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
}
#endif
