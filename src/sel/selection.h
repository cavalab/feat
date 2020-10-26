/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H

#include "selection_operator.h"
#include "lexicase.h"
#include "fair_lexicase.h"
#include "nsga2.h"
#include "tournament.h"
#include "offspring.h"
#include "random.h"
#include "simulated_annealing.h"

namespace FT{

    
    struct Parameters; // forward declaration of Parameters      
    
    /**
     * @namespace FT::Sel
     * @brief namespace containing Selection methods for best individuals 
     * used in Feat
     */
    namespace Sel{ 
        ////////////////////////////////////////////////////////// Declarations
	
        /*!
         * @class Selection
         * @brief interfaces with selection operators. 
         */
        struct Selection
        {
            shared_ptr<SelectionOperator> pselector; 
            string type;
            bool survival;
            
            Selection(string type="lexicase", bool survival=false);
            
            ~Selection();

            void set_operator();
            
            /// return type of selectionoperator
            string get_type();
            void set_type(string);
            
            /// perform selection 
            vector<size_t> select(Population& pop,  
                    const Parameters& params, const Data& d);
            
            /// perform survival
            vector<size_t> survive(Population& pop,  
                    const Parameters& params, const Data& d);
        };

        NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Selection, type, survival);
        
    }
    
}
#endif
