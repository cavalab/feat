/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "selection.h"

namespace FT{

    namespace SelectionSpace{
           
        Selection::Selection(string type, bool survival)
        {
            /*!
             * set type of selection operator.
             */

            if (!type.compare("lexicase"))
                pselector = std::make_shared<Lexicase>(survival); 
            else if (!type.compare("nsga2"))
                pselector = std::make_shared<NSGA2>(survival);
            else if (!type.compare("offspring"))    // offspring survival
                pselector = std::make_shared<Offspring>(survival);
            else if (!type.compare("random"))    // offspring survival
                pselector = std::make_shared<Random>(survival);
            else if (!type.compare("simanneal"))    // offspring survival
                pselector = std::make_shared<SimAnneal>(survival);
            else
                HANDLE_ERROR_NO_THROW("Undefined Selection Operator " + type + "\n");
                
        }

        Selection::~Selection(){}
        
        /// return type of selectionoperator
        string Selection::get_type(){ return pselector->name; }
        
        /// perform selection 
        vector<size_t> Selection::select(Population& pop, const MatrixXd& F, const Parameters& params)
        {       
            return pselector->select(pop, F, params);
        }
        /// perform survival
        vector<size_t> Selection::survive(Population& pop, const MatrixXd& F,  const Parameters& params)
        {       
            return pselector->survive(pop, F, params);
        }
    }
    
}

