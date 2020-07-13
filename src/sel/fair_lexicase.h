/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FAIRLEXICASE_H
#define FAIRLEXICASE_H

#include "selection_operator.h"


namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////// Declarations
        /*!
         * @class FairLexicase
         * @brief FairLexicase selection operator.
         */
        struct FairLexicase : SelectionOperator
        {
            FairLexicase(bool surv);
            
            ~FairLexicase();

            /// function returns a set of selected indices  
            vector<size_t> select(Population& pop,  
                    const Parameters& params, const Data& d); 
            
            /// lexicase survival
            vector<size_t> survive(Population& pop,  
                    const Parameters& params, const Data& d); 

        };
    }

}

#endif
