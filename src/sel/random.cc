/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "random.h"

namespace FT{

    namespace Sel{

        /** Random based selection and survival methods. */

        Random::Random(bool surv){ name = "random"; survival = surv; elitism = true;};
        
        Random::~Random(){}
           
        vector<size_t> Random::select(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selects parents for making offspring.  
             *
             * @param pop: population of programs, all potential parents. 
             * @param params: parameters.
             *
             * @return selected: vector of indices corresponding to offspring that are selected.
             *      
             */
          
            int P = pop.size(); // index P is where the offspring begin, and also the size of the pop
           
            vector<size_t> all_idx(pop.size());
            std::iota(all_idx.begin(), all_idx.end(), 0);

            cout << "selecting randoms\n";       
            vector<size_t> selected;
            for (unsigned i = 0; i < P; ++i)    
                selected.push_back(r.random_choice(all_idx));   // select randomly
           
            cout << "getting elite\n";       
            if (elitism)
                enforce_elite(pop, selected); 
            
            return selected;
        }

        vector<size_t> Random::survive(Population& pop, 
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
           
            vector<size_t> all_idx(pop.size());
            std::iota(all_idx.begin(), all_idx.end(), 0);

            cout << "selecting randoms\n";       
            vector<size_t> selected;
            for (unsigned i = 0; i < P; ++i)    
                selected.push_back(r.random_choice(all_idx));   // select randomly
            cout << "getting elite\n";       
            if (elitism)
                enforce_elite(pop, selected); 
            
            return selected;
        }
       
        void Random::enforce_elite(Population& pop, vector<size_t>& selected)
        {
            // find best and worst inds and if best is not in selected, replace worst with it
            size_t best_idx, worst_idx;
            float min_fit, max_fit;

            for (unsigned i = 0; i < pop.individuals.size(); ++i)
            {
                if (pop.individuals.at(i).fitness < min_fit || i == 0)
                {
                    min_fit = pop.individuals.at(i).fitness;
                    best_idx = i;
                }
            }    
            
            if (!in(selected, best_idx) )   // add best individual to selected
            {
                for (unsigned i = 0; i < selected.size(); ++i)
                {
                    if (pop.individuals.at(selected.at(i)).fitness > max_fit || i == 0)
                    {
                        max_fit = pop.individuals.at(selected.at(i)).fitness;
                        worst_idx = i;
                    }
                    
                }
                selected.at(worst_idx) = best_idx;
            }
            
        }
    }

}
