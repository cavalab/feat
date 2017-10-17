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
        /*!
         * Pareto selection operator.
         */
        Pareto(bool surv){ survival = surv; };
        
        ~Pareto(){}

        vector<size_t> select(const MatrixXd& F, const Parameters& p, Rnd& r){};

        private:
            vector<vector<int>> front;                      //< the Pareto fronts
            void fast_nds(Population&);               //< Fast non-dominated sorting 
            void crowding_distance(Population&, int); //< crowding distance of a front 
    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions

    vector<size_t> select(Population& pop, const Parameters& params, Rnd& r)
    {
        /* Selection using the survival scheme of NSGA-II. 
         *
         * Input: 
         *
         *      F: n_samples X popsize matrix of model outputs.
         *      params: parameters.
         *
         * Output:
         *
         *      selected: vector of indices corresponding to columns of F that are selected.
         */
        
        // set objectives
        for (const auto& p : pop)
            p.set_objectives(params.objectives);

        // fast non-dominated sort
        fast_nds(pop);
        
        // Push back selected individuals until full
        vector<size_t> selected;
        while ( selected.size() + front[i].size() < params.popsize)
        {
            std::vector<int>& Fi = front[i];        // indices in front i
            //crowding_distance(i);                   // calculate crowding in Fi

            for (int j = 0; j < Fi.size(); ++j)     // Pt+1 = Pt+1 U Fi
                selected.push_back(Fi[j]);
            
            ++i;
        }

        crowding_distance(i);   // calculate crowding in final front to include
        std::sort(front[i].begin(),front[i].end(),sort_n());

        const int extra = popsize - selected.size();
        for (int j = 0; j < extra; ++j) // Pt+1 = Pt+1 U Fi[1:N-|Pt+1|]
            selected.push_back(front[i][j]);

        return selected;
    }

    void Pareto::fast_nds(Population& pop) 
    {
        front.resize(1);
        front[0].clear();
        //std::vector< std::vector<int> >  F(1);
        #pragma omp parallel for
        for (int i = 0; i < pop.size(); ++i) {
        
            std::vector<int> dom;
            int dcount = 0;
        
            individual& p = pop[i];
            // p.dcounter  = 0;
            // p.dominated.clear();
        
            for (int j = 0; j < pop.size(); ++j) {
            
                individual& q = pop[j];
            
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
                    p.rank = 1;
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

                individual& p = pop[fronti[i]];

                for (int j = 0; j < p.dominated.size() ; ++j) {

                    individual& q = pop[p.dominated[j]];
                    q.dcounter -= 1;

                    if (q.dcounter == 0) {
                        q.rank = fi+1;
                        Q.push_back(p.dominated[j]);
                    }
                }
            }

            fi += 1;
            front.push_back(Q);
        }

    }

    void Pareto::crowding_distance(const Population& pop, int fronti)
    {
        std::vector<int> F = front[fronti];
        if (F.size() == 0 ) return;

        const int fsize = F.size();

        for (int i = 0; i < fsize; ++i)
            pop[F[i]].crowd_dist = 0;
   

        const int limit = pop[0].obj.size();
        for (int m = 0; m < limit; ++m) {

            std::sort(F.begin(), F.end(), comparator_obj(*this,m));

            // in the paper dist=INF for the first and last, in the code
            // this is only done to the first one or to the two first when size=2
            pop[F[0]].crowd_dist = INF;
            if (fsize > 1)
                pop[F[fsize-1]].crowd_dist = INF;
        
            for (int i = 1; i < fsize-1; ++i) 
            {
                if (pop[F[i]].crowd_dist != INF) 
                {
                    pop[F[i]].crowd_dist +=
                        (pop[F[i+1]].obj[m] - pop[F[i-1]].obj[m]) // crowd over obj
                        / (pop[F[fsize-1]].obj[m] - pop[F[0]].obj[m]);
                }
            }
        }        
    }
    
    struct sort_n {
        const population& pop;
        sort_n(const population& population) : pop(population) {};
        bool operator() (int i, int j) {
            const individual& ind1 = pop.ind[i];
            const individual& ind2 = pop.ind[j];
            if (ind1.rank < ind2.rank)
                return true;
            else if (ind1.rank == ind2.rank &&
                     ind1.crowd_dist > ind2.crowd_dist)
                return true;
            return false;
        };
    };

    struct comparator_obj 
    {
        comparator_obj(const Population& population, int index) :
            pop(population), m(index) {};
        const population& pop;
        int m;
        bool operator() (int i, int j) { return pop[i].obj[m] < pop[j].obj[m]; };
    };
}
#endif
