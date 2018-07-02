/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef SIMANNEAL_H
#define SIMANNEAL_H

#include "selection_operator.h"

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class SimAnneal
     */
    struct SimAnneal : SelectionOperator
    {
        /** SimAnneal based selection and survival methods. */

        SimAnneal(bool surv);
        
        ~SimAnneal();
       
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params);
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params);
    private:
        double t;           ///< annealing temperature
        double t0;          ///< initial temperature
    };
    
}
#endif
