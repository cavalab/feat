/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FAIRLEXICASE2_H
#define FAIRLEXICASE2_H

#include "selection_operator.h"


namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class FairLexicase2
         * @brief FairLexicase2 selection operator.
         */
        struct FairLexicase2 : SelectionOperator
        {
            FairLexicase2(bool surv);
            
            ~FairLexicase2();

            /// function returns a set of selected indices from F. 
            vector<size_t> select(Population& pop, const MatrixXf& F, 
                    const Parameters& params, const Data& d); 
            
            /// lexicase survival
            vector<size_t> survive(Population& pop, const MatrixXf& F, 
                    const Parameters& params, const Data& d); 

        };
    }

}

#endif
