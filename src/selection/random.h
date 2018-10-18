/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef RANDOM_H
#define RANDOM_H

#include "selection_operator.h"

namespace FT{

    namespace SelectionSpace{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Random
         */
        struct Random : SelectionOperator
        {
            /** Random based selection and survival methods. */

            Random(bool surv);
            
            ~Random();
           
            vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params);
            vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params);
            /// replaces worst individual in selected with best individual in Pop.
            void enforce_elite(Population& pop, vector<size_t>& selected);
            bool elitism;       //< whether or not to keep the best individual.

        };
        
    }
    
}
#endif
