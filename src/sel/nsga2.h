/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARETO_H
#define PARETO_H

#include "selection_operator.h"

namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class NSGA2
         */
        struct NSGA2 : SelectionOperator
        {
            /** NSGA-II based selection and survival methods. */

            NSGA2(bool surv);
            
            ~NSGA2();

            /// selection according to the survival scheme of NSGA-II
            vector<size_t> select(Population& pop,  
                    const Parameters& p, const Data& d);
            
            /// survival according to the survival scheme of NSGA-II
            vector<size_t> survive(Population& pop,  
                    const Parameters& p, const Data& d);
            
            //< the Pareto fronts
            vector<vector<int>> front;                

            //< Fast non-dominated sorting
            void fast_nds(vector<Individual>&);                

            //< crowding distance of a front i
            void crowding_distance(Population&, int); 
                
            private:    

                /// sort based on rank, breaking ties with crowding distance
                struct sort_n 
                {
                    const Population& pop;          ///< population address
                    sort_n(const Population& population) : pop(population) {};
                    bool operator() (int i, int j) {
                        const Individual& ind1 = pop.individuals[i];
                        const Individual& ind2 = pop.individuals[j];
                        if (ind1.rank < ind2.rank)
                            return true;
                        else if (ind1.rank == ind2.rank &&
                                 ind1.crowd_dist > ind2.crowd_dist)
                            return true;
                        return false;
                    };
                };

                /// sort based on objective m
                struct comparator_obj 
                {
                    const Population& pop;      ///< population address
                    int m;                      ///< objective index 
                    comparator_obj(const Population& population, int index) 
                        : pop(population), m(index) {};
                    bool operator() (int i, int j) { return pop[i].obj[m] < pop[j].obj[m]; };
                };
            
                size_t tournament(vector<Individual>& pop, size_t i, size_t j) const;
        };
        
    }
    
}
#endif
