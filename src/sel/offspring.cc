/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "offspring.h"

namespace FT{

    namespace Sel{

        Offspring::Offspring(bool surv){ name = "offspring"; survival = surv; elitism = true;};
        
        Offspring::~Offspring(){}
           
        vector<size_t> Offspring::survive(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selects the offspring for survival. 
             *
             * @param pop: population of programs, parents + offspring.
             * @param params: parameters.
             *
             * @return selected: vector of indices corresponding to offspring that are selected.
             *      
             */
          
            int P = pop.individuals.size()/2; // index P is where the offspring begin, and also the size of the pop
            
            vector<size_t> selected(P);
            // select popsize/2 to popsize individuals
            std::iota(selected.begin(),selected.end(),P);
            
            if (selected.at(selected.size()-1) > pop.size())
                THROW_LENGTH_ERROR("error: selected includes " +
                                    to_string(selected.at(selected.size()-1)) +
                                    ", pop size is " + to_string(pop.size()) + "\n");
              
            if (elitism)
            {   // find best and worst inds and if best is not in selected, replace worst with it
                size_t best_idx, worst_idx;
                float min_fit, max_fit;

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
}
