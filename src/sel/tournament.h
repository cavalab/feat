/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef TOURNAMENT_H
#define TOURNAMENT_H

#include "selection_operator.h"

namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Tournament
         */
        struct Tournament : SelectionOperator
        {
            /** Tournament based selection and survival methods. */

            Tournament(bool surv);
            
            ~Tournament();

            /// selection according to the survival scheme of Tournament
            vector<size_t> select(Population& pop,  
                    const Parameters& p, const Data& d);
            
            /// survival according to the survival scheme of Tournament
            vector<size_t> survive(Population& pop, 
                    const Parameters& p, const Data& d);

            private:    

                size_t tournament(vector<Individual>& pop, size_t i, size_t j) const;
        };
        
    }
    
}
#endif
