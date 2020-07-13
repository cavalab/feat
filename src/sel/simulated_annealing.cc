/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "simulated_annealing.h"

namespace FT{

    namespace Sel{

        SimAnneal::SimAnneal(bool surv){ name = "simanneal"; survival = surv; t0 = 10;};
        
        SimAnneal::~SimAnneal(){}
           
        vector<size_t> SimAnneal::select(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selects parents for making offspring.  
             *
             * @param pop: population of programs, all potential parents
             * @param params: parameters.
             *
             * @return selected: vector of indices corresponding to offspring that are selected.
             *      
             */
          
            int P = pop.size(); // index P is where the offspring begin, and also the size of the pop
           
            vector<size_t> all_idx(pop.size());
            std::iota(all_idx.begin(), all_idx.end(), 0);
            return all_idx;
        }

        vector<size_t> SimAnneal::survive(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selects the offspring for survival using simulated annealing.
             *
             * Offspring are compared to their parents. The probability of an offspring, R, replacing 
             * its parent, S, is given by
             * 
             *      P(t, R, S ) = exp( ( fitness(S) - fitness(R) ) / t)
             *
             * where t is the temperature. 
             *
             * @param pop: population of programs, parents + offspring.
             * @param params: parameters.
             *
             * @return selected: vector of indices corresponding to offspring that are selected.
             *      
             */
         
            // decrease temperature linearly based on current generation
            // cooling schedule: Tg = (0.9)^g * t0, g = current generation
            this->t = pow(0.9, float(params.current_gen))*this->t0;  
            /* cout << "t: " << this->t << "\n"; */

            int P = pop.individuals.size()/2; // index P is where the offspring begin, and also the size of the pop
            vector<size_t> selected(P); 
            #pragma omp parallel for        
            for (unsigned i = P; i < pop.individuals.size(); ++i)
            {
                Individual& offspring = pop.individuals.at(i);
                /* cout << "offspring: " << offspring.get_eqn() << "\n"; */
                int pid = offspring.parent_id.at(0);
                bool found = false;
                int j = 0;
                while (!found && j < P)
                {
                    if ( pop.individuals.at(j).id == pid)
                        found=true;
                    else
                        ++j;
                }
                Individual& parent = pop.individuals.at(j);
                /* cout << "parent: " << parent.get_eqn() << "\n"; */
                /* cout << "offspring fitness: " << offspring.fitness << "\n"; */
                /* cout << "parent fitness: " << parent.fitness << "\n"; */           
                float probability = exp ( (parent.fitness - offspring.fitness)/this->t );

                /* cout << "probability: " << probability << "\n"; */
                if (r() < probability)
                    selected.at(i-P) = i;
                else
                    selected.at(i-P) = j;
            }
            return selected;
        }
    }
}

