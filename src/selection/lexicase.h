/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef LEXICASE_H
#define LEXICASE_H

#include "selection_operator.h"


namespace FT{

    namespace SelectionSpace{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Lexicase
         * @brief Lexicase selection operator.
         */
        struct Lexicase : SelectionOperator
        {
            Lexicase(bool surv);
            
            ~Lexicase();

            /// function returns a set of selected indices from F. 
            vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params); 
            
            /// lexicase survival
            vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params); 

        };
    }

}

#endif
