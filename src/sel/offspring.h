/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef OFFSPRING_H
#define OFFSPRING_H

#include "selection_operator.h"

namespace FT{
    
    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Offspring
         */
        struct Offspring : SelectionOperator
        {
            /** Offspring based selection and survival methods. */

            Offspring(bool surv);
            
            ~Offspring();
           
            vector<size_t> survive(Population& pop,  
                    const Parameters& params, const Data& d);

            bool elitism;       //< whether or not to keep the best individual.

        };
    }
}
#endif
