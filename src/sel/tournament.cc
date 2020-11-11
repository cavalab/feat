/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "tournament.h"

namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Tournament
         */
        
        Tournament::Tournament(bool surv){ name = "nsga2"; survival = surv; };
        
        Tournament::~Tournament(){}
        
        size_t Tournament::tournament(vector<Individual>& pop, size_t i, size_t j) const 
        {
            Individual& ind1 = pop.at(i);
            Individual& ind2 = pop.at(j);

            if (ind1.fitness < ind2.fitness)
                return i;
            else if (ind1.fitness == ind2.fitness)
                return r() < 0.5 ? i : j;
            else
                return j;

        }
        
        vector<size_t> Tournament::select(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selection using tournament selection. 
             *
             * Input: 
             *
             *      pop: population of programs.
             *      params: parameters.
             *      r: random number generator
             *
             * Output:
             *
             *      selected: vector of indices corresponding to pop that are selected.
             *      modifies individual ranks, objectives and dominations.
             */
            vector<size_t> pool(pop.size());
            std::iota(pool.begin(), pool.end(), 0);
            // if this is first generation, just return indices to pop
            if (params.current_gen==0)
                return pool;

            vector<size_t> selected(pop.size());

            for (int i = 0; i < pop.size(); ++i)
            {
                size_t winner = tournament(pop.individuals, r.random_choice(pool), 
                                           r.random_choice(pool));
                selected.push_back(winner);
            }
            return selected;
        }

        vector<size_t> Tournament::survive(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selection using the survival scheme of NSGA-II. 
             *
             * Input: 
             *
             *      pop: population of programs.
             *      params: parameters.
             *      r: random number generator
             *
             * Output:
             *
             *      selected: vector of indices corresponding to pop that are selected.
             *      modifies individual ranks, objectives and dominations.
             */
            
            THROW_RUNTIME_ERROR("Not implemented");
            return vector<size_t>();
        }


    }
    
}

