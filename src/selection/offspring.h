/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef OFFSPRING_H
#define OFFSPRING_H

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Offspring
     */
    struct Offspring : SelectionOperator
    {
        /** Offspring based selection and survival methods. */

        Offspring(bool surv){ name = "offspring"; survival = surv; elitism = true;};
        
        ~Offspring(){}
       
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params);

        bool elitism;       //< whether or not to keep the best individual.

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions

    vector<size_t> Offspring::survive(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Selects the offspring for survival. 
         *
         * @params pop: population of programs, parents + offspring.
         * @params F: n_samples x 2 * popsize matrix of program behaviors. 
         * @params params: parameters.
         *
         * @returns selected: vector of indices corresponding to offspring that are selected.
         *      
         */
      
        int P = F.cols()/2; // index P is where the offspring begin, and also the size of the pop
        
        vector<size_t> selected(P);
        // select F/2 to F individuals
        std::iota(selected.begin(),selected.end(),P);
        if (selected[selected.size()-1] > pop.size())
        {
            cout << "error: selected includes " << selected.at(selected.size()-1) 
                << ", pop size is " << pop.size() << "\n";
            exit(1);
        }
        if (elitism)
        {   // find best and worst inds and if best is not in selected, replace worst with it
            size_t best_idx, worst_idx;
            double min_fit, max_fit;

            for (unsigned i = 0; i < pop.individuals.size(); ++i)
            {
                if (pop.individuals.at(i).fitness < min_fit || i == 0)
                {
                    min_fit = pop.individuals.at(i).fitness;
                    best_idx = i;
                }
                if (i >= P)  // finds worst among offspring
                {
                    if (pop.individuals.at(i).fitness > max_fit || i == P)
                    {
                        max_fit = pop.individuals.at(i).fitness;
                        worst_idx = i;
                    }
                }
            }
            if (best_idx < P)   // then best individual is in parents, so replace worst_idx
            {
                selected.at(worst_idx - P) = best_idx;
            }
        }
        return selected;
    }
    
    
}
#endif
