/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_OPERATOR_H
#define SELECTION_OPERATOR_H

#include "../pop/population.h"

namespace FT{

    namespace SelectionSpace{

        /*!
         * @class SelectionOperator
         * @brief base class for selection operators.
         */ 
        struct SelectionOperator 
        {
            bool survival; 
            string name;

            //SelectionOperator(){}

            virtual ~SelectionOperator();
            
            virtual vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& p);
            
            virtual vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& p);
        };
    }
}

#endif
