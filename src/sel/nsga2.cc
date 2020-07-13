/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nsga2.h"

namespace FT{

    namespace Sel{
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class NSGA2
         */
        
        NSGA2::NSGA2(bool surv){ name = "nsga2"; survival = surv; };
        
        NSGA2::~NSGA2(){}
        
        size_t NSGA2::tournament(vector<Individual>& pop, size_t i, size_t j) const 
        {
            Individual& ind1 = pop.at(i);
            Individual& ind2 = pop.at(j);

            int flag = ind1.check_dominance(ind2);
            
            if (flag == 1) // ind1 dominates ind2
                return i;
            else if (flag == -1) // ind2 dominates ind1
                return j;
            else if (ind1.crowd_dist > ind2.crowd_dist)
                return i;
            else if (ind2.crowd_dist > ind1.crowd_dist)
                return j;
            else 
                return i; 
        }
        
        vector<size_t> NSGA2::select(Population& pop,  
                const Parameters& params, const Data& d)
        {
            /* Selection using Pareto tournaments. 
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

        vector<size_t> NSGA2::survive(Population& pop, 
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
            
            // set objectives
            #pragma omp parallel for
            for (unsigned int i=0; i<pop.size(); ++i)
                pop.individuals.at(i).set_obj(params.objectives);

            // fast non-dominated sort
            fast_nds(pop.individuals);
            
            // Push back selected individuals until full
            vector<size_t> selected;
            int i = 0;
            while ( selected.size() + front.at(i).size() < params.pop_size)
            {
                std::vector<int>& Fi = front.at(i);        // indices in front i
                crowding_distance(pop,i);                   // calculate crowding in Fi

                for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
                    selected.push_back(Fi.at(j));
                
                ++i;
            }

            crowding_distance(pop,i);   // calculate crowding in final front to include
            std::sort(front.at(i).begin(),front.at(i).end(),sort_n(pop));

            const int extra = params.pop_size - selected.size();
            for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
                selected.push_back(front.at(i).at(j));
            
            return selected;
        }

        void NSGA2::fast_nds(vector<Individual>& individuals) 
        {
            front.resize(1);
            front.at(0).clear();
            //std::vector< std::vector<int> >  F(1);
            #pragma omp parallel for
            for (int i = 0; i < individuals.size(); ++i) {
            
                std::vector<unsigned int> dom;
                int dcount = 0;
            
                Individual& p = individuals.at(i);
                // p.dcounter  = 0;
                // p.dominated.clear();
            
                for (int j = 0; j < individuals.size(); ++j) {
                
                    Individual& q = individuals.at(j);
                
                    int compare = p.check_dominance(q);
                    if (compare == 1) { // p dominates q
                        //p.dominated.push_back(j);
                        dom.push_back(j);
                    } else if (compare == -1) { // q dominates p
                        //p.dcounter += 1;
                        dcount += 1;
                    }
                }
            
                #pragma omp critical
                {
                    p.dcounter  = dcount;
                    p.dominated.clear();
                    p.dominated = dom;
                
                
                    if (p.dcounter == 0) {
                        p.set_rank(1);
                        front.at(0).push_back(i);
                    }
                }
            
            }
            
            // using OpenMP can have different orders in the front.at(0)
            // so let's sort it so that the algorithm is deterministic
            // given a seed
            std::sort(front.at(0).begin(), front.at(0).end());    

            int fi = 1;
            while (front.at(fi-1).size() > 0) {

                std::vector<int>& fronti = front.at(fi-1);
                std::vector<int> Q;
                for (int i = 0; i < fronti.size(); ++i) {

                    Individual& p = individuals.at(fronti.at(i));

                    for (int j = 0; j < p.dominated.size() ; ++j) {

                        Individual& q = individuals.at(p.dominated.at(j));
                        q.dcounter -= 1;

                        if (q.dcounter == 0) {
                            q.set_rank(fi+1);
                            Q.push_back(p.dominated.at(j));
                        }
                    }
                }

                fi += 1;
                front.push_back(Q);
            }

        }

        void NSGA2::crowding_distance(Population& pop, int fronti)
        {
            std::vector<int> F = front.at(fronti);
            if (F.size() == 0 ) return;

            const int fsize = F.size();

            for (int i = 0; i < fsize; ++i)
                pop.individuals.at(F.at(i)).crowd_dist = 0;
       

            const int limit = pop.individuals.at(0).obj.size();
            for (int m = 0; m < limit; ++m) {

                std::sort(F.begin(), F.end(), comparator_obj(pop,m));

                // in the paper dist=INF for the first and last, in the code
                // this is only done to the first one or to the two first when size=2
                pop.individuals.at(F.at(0)).crowd_dist = std::numeric_limits<float>::max();
                if (fsize > 1)
                    pop.individuals.at(F.at(fsize-1)).crowd_dist = std::numeric_limits<float>::max();
            
                for (int i = 1; i < fsize-1; ++i) 
                {
                    if (pop.individuals.at(F.at(i)).crowd_dist != std::numeric_limits<float>::max()) 
                    {   // crowd over obj
                        pop.individuals.at(F.at(i)).crowd_dist +=
                            (pop.individuals.at(F.at(i+1)).obj.at(m) - pop.individuals.at(F.at(i-1)).obj.at(m)) 
                            / (pop.individuals.at(F.at(fsize-1)).obj.at(m) - pop.individuals.at(F.at(0)).obj.at(m));
                    }
                }
            }        
        }
    }
    
}

