/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARETO_H
#define PARETO_H

namespace FT{
/*!
     * @class Pareto
     */
    struct Pareto : SelectionOperator
    {
        /*!
         * Pareto selection operator.
         */
        Pareto(bool surv){ survival = surv; };
        
        ~Pareto(){}

        vector<size_t> select(const MatrixXd& F, const Parameters& p, Rnd& r){};

    };
}
#endif
