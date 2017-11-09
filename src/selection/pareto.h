/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARETO_H
#define PARETO_H

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Pareto
     */
    struct Pareto : SelectionOperator
    {
        /** NSGA-II based selection and survival methods. */

        Pareto(bool surv){ name = "pareto"; survival = surv; };
        
        ~Pareto(){}

        /// selection according to the survival scheme of NSGA-II
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& p);

        private:
            vector<vector<int>> front;                //< the Pareto fronts
            void fast_nds(Population&);               //< Fast non-dominated sorting 
            void crowding_distance(Population&, int); //< crowding distance of a front i
            
           
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
    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions

    bool check_pareto(Population& pop, vector<size_t> survivors)
    {
        /// check to make sure nsga-ii is returning rank=1 candidates and never missing the front. 
        
        // get ranks of the population
        vector<unsigned> ranks, s_ranks;
        unsigned front_num=0, s_front_num=0;
        for (const auto& p: pop.individuals)
        {
            ranks.push_back(p.rank);
            if (p.rank==1) ++front_num;
        }
        for (const auto& s: survivors){
            s_ranks.push_back(pop.individuals[s].rank);
            if (pop.individuals[s].rank==1) ++ s_front_num;
        }
        // check that all individuals with rank=1 are in survivors, if there are less than popsize
        //std::cout << "Pareto front count in pop+offspring: " << front_num << "\n";
        //std::cout << "Pareto front count in survivors: " << s_front_num << "\n";
        return s_front_num == front_num || front_num > survivors.size();
    }

    vector<size_t> Pareto::survive(Population& pop, const MatrixXd& F, const Parameters& params)
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
         *      selected: vector of indices corresponding to columns of F that are selected.
         *      modifies individual ranks, objectives and dominations.
         */
        
        // set objectives
        #pragma omp parallel for
        for (unsigned int i=0; i<pop.size(); ++i)
            pop.individuals[i].set_obj(params.objectives);

        // fast non-dominated sort
        fast_nds(pop);
        
        // Push back selected individuals until full
        vector<size_t> selected;
        int i = 0;
        while ( selected.size() + front[i].size() < params.pop_size)
        {
            std::vector<int>& Fi = front[i];        // indices in front i
            //crowding_distance(i);                   // calculate crowding in Fi

            for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
                selected.push_back(Fi[j]);
            
            ++i;
        }

        crowding_distance(pop,i);   // calculate crowding in final front to include
        std::sort(front[i].begin(),front[i].end(),sort_n(pop));

        const int extra = params.pop_size - selected.size();
        for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
            selected.push_back(front[i][j]);
        
        assert(check_pareto(pop,selected));
        
        return selected;
    }

    void Pareto::fast_nds(Population& pop) 
    {
        front.resize(1);
        front[0].clear();
        //std::vector< std::vector<int> >  F(1);
        #pragma omp parallel for
        for (int i = 0; i < pop.size(); ++i) {
        
            std::vector<unsigned int> dom;
            int dcount = 0;
        
            Individual& p = pop.individuals[i];
            // p.dcounter  = 0;
            // p.dominated.clear();
        
            for (int j = 0; j < pop.size(); ++j) {
            
                Individual& q = pop.individuals[j];
            
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
                    front[0].push_back(i);
                }
            }
        
        }
        
        // using OpenMP can have different orders in the front[0]
        // so let's sort it so that the algorithm is deterministic
        // given a seed
        std::sort(front[0].begin(), front[0].end());    

        int fi = 1;
        while (front[fi-1].size() > 0) {

            std::vector<int>& fronti = front[fi-1];
            std::vector<int> Q;
            for (int i = 0; i < fronti.size(); ++i) {

                Individual& p = pop.individuals[fronti[i]];

                for (int j = 0; j < p.dominated.size() ; ++j) {

                    Individual& q = pop.individuals[p.dominated[j]];
                    q.dcounter -= 1;

                    if (q.dcounter == 0) {
                        q.set_rank(fi+1);
                        Q.push_back(p.dominated[j]);
                    }
                }
            }

            fi += 1;
            front.push_back(Q);
        }

    }

    void Pareto::crowding_distance(Population& pop, int fronti)
    {
        std::vector<int> F = front[fronti];
        if (F.size() == 0 ) return;

        const int fsize = F.size();

        for (int i = 0; i < fsize; ++i)
            pop.individuals[F[i]].crowd_dist = 0;
   

        const int limit = pop[0].obj.size();
        for (int m = 0; m < limit; ++m) {

            std::sort(F.begin(), F.end(), comparator_obj(pop,m));

            // in the paper dist=INF for the first and last, in the code
            // this is only done to the first one or to the two first when size=2
            pop.individuals[F[0]].crowd_dist = std::numeric_limits<double>::max();
            if (fsize > 1)
                pop.individuals[F[fsize-1]].crowd_dist = std::numeric_limits<double>::max();
        
            for (int i = 1; i < fsize-1; ++i) 
            {
                if (pop.individuals[F[i]].crowd_dist != std::numeric_limits<double>::max()) 
                {   // crowd over obj
                    pop.individuals[F[i]].crowd_dist +=
                        (pop.individuals[F[i+1]].obj[m] - pop.individuals[F[i-1]].obj[m]) 
                        / (pop.individuals[F[fsize-1]].obj[m] - pop.individuals[F[0]].obj[m]);
                }
            }
        }        
    }
    
    
}
#endif
