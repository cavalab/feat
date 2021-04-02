/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "fair_lexicase.h"

namespace FT{

namespace Sel{

FairLexicase::FairLexicase(bool surv)
{ 
    name = "fair_lexicase"; 
    survival = surv; 
}

FairLexicase::~FairLexicase(){}

vector<size_t> FairLexicase::select(Population& pop,  
        const Parameters& params, const Data& d)
{
    /*! Selection according to lexicase selection for 
     * binary outcomes and epsilon-lexicase selection for continuous. 
     * @param pop: population
     * @param params: parameters.
     *
     * @return selected: vector of indices corresponding to pop that 
     * are selected.
     *
     */            

    //< number of samples
    unsigned int N = pop.individuals.at(0).error.size(); 
    //< number of individuals
    unsigned int P = pop.individuals.size();             
    
    vector<size_t> starting_pool;
    for (int i = 0; i < pop.individuals.size(); ++i)
    {
        starting_pool.push_back(i);
    }
    assert(starting_pool.size() == P);     

    // selected individuals
    vector<size_t> selected(P,0); 
    #pragma omp parallel for 
    for (unsigned int i = 0; i<P; ++i)  // selection loop
    {
        vector<size_t> pool = starting_pool;    // initial pool   
        vector<size_t> winner;                  // winners


        vector<size_t> shuff_idx; // used to shuffle cases
        map<int,vector<float>> protect_levels;
        vector<float> used_levels;
        if (!d.cases.empty())
        {
            /* cout << "d.cases is not empty; using...\n"; */
            shuff_idx.resize(d.cases.size()); 
            std::iota(shuff_idx.begin(),shuff_idx.end(),0);
            // shuffle cases
            r.shuffle(shuff_idx.begin(),shuff_idx.end());   
        }
        else
        {
            // store a local copy of protect levels to prevent repeats
            if (d.protect_levels.size() == 0)
            {
                /* cout << "d.protect_levels empty"; */
            }
            protect_levels = d.protect_levels;
        }


        bool pass = true;     // checks pool size and number of cases
        unsigned int h = 0;   // case count

        while(pass){    // main loop

            // **Fairness Subgroups**
            // Here, a "case" is the mean fitness over a  collection of
            // samples sharing an intersection of levels of
            // protected groups. 
            
            // if the cases haven't been enumerated, 
            // we sample group intersections.
            ArrayXb x_idx; 
            if (d.cases.empty())                        
            {
                /* cout << "d.cases is empty, so sampling group " */ 
                /*     << "intersections\n"; */
                /* cout << "current protect_levels:\n"; */
                /* for (auto pl : protect_levels) */
                /* { */
                /*     cout << "\tfeature " << pl.first << ":" */
                /*         << pl.second.size() << " values; "; */    
                /* } */
                /* cout << "\n"; */
                vector<int> groups;
                // push back current groups (keys)
                for (auto pl : protect_levels)
                {
                    groups.push_back(pl.first);
                }
                x_idx = ArrayXb::Constant(d.X.cols(),true);
                /* cout << "x_idx sum: " << x_idx.count() << "\n"; */
                // choose a random group
                vector<size_t> choice_idxs(groups.size());
                std::iota(choice_idxs.begin(),choice_idxs.end(),0);
                size_t idx = r.random_choice(choice_idxs);
                int g = groups.at(idx); 
                /* cout << "chosen group: " << g << "\n"; */
                // choose a random level
                vector<float> lvls = unique(VectorXf(d.X.row(g)));
                // remove levels not in protect_levels[g]
                for (int i = lvls.size()-1; i --> 0; )
                {
                    // ^ that's a backwards loop alright!
                    if (!in(protect_levels.at(g),lvls.at(i)))
                        lvls.erase(lvls.begin() + i);
                }
                float level = r.random_choice(lvls);
                /* cout << "chosen level: " << level << "\n"; */
                // remove chosen level from protect_levels
                for (int i = 0; i < protect_levels.at(g).size(); ++i)
                {
                    if (protect_levels[g].at(i) == level)
                    {
                        protect_levels[g].erase(
                                protect_levels[g].begin() + i);
                        break;
                    }
                }

                x_idx = (d.X.row(g).array() == level);
                /* cout << "x_idx count: " << x_idx.count() << "\n"; */
            }
            const ArrayXb& case_idx = (d.cases.empty() ? 
                                x_idx : d.cases.at(shuff_idx[h]));
            // get fitness of everyone in the pool
            ArrayXf fitness(pool.size());
            for (size_t j = 0; j<pool.size(); ++j)
            {
                if (r() < 0.5)
                {
                    // half the time use loss
                    fitness(j) = case_idx.select(
                            pop.individuals.at(pool[j]).error,
                            0).sum();
                }
                else
                {
                    // half the time, look at fairness
                    fitness(j) = fabs(pop.individuals.at(pool[j]).error.sum()
                                      - case_idx.select(
                                        pop.individuals.at(pool[j]).error,
                                        0).sum());
                }
            }
            // get epsilon for the fitnesses
            float epsilon = mad(fitness);
            //
            winner.resize(0);   // winners                  
            // minimum error on case
            float minfit = std::numeric_limits<float>::max();                     

            // get minimum
            for (size_t j = 0; j<pool.size(); ++j)
            {
                if (fitness(j) < minfit) 
                  minfit = fitness(j);
            }

            // select best
            for (size_t j = 0; j<pool.size(); ++j)
            {
                if (fitness(j) <= minfit+epsilon)
                    winner.push_back(pool[j]);                 
            }

            /* cout << "h: " << h << endl; */
            /* cout << "winner size: " << winner.size() << "\n"; */
            ++h; // next case
            // only keep going if needed
            pass = (winner.size()>1 
                   && h<shuff_idx.size()); 

            if(winner.size() == 0)
            {
                if(h >= d.group_intersections)
                    winner.push_back(r.random_choice(pool));
                else
                    pass = true;
            }
            else
            {
                // reduce pool to remaining individuals
                pool = winner;    
            }
      
        }       
    
        assert(winner.size()>0);
        //if more than one winner, pick randomly
        selected.at(i) = r.random_choice(winner);   
    }               

    if (selected.size() != pop.individuals.size())
    {
        std::cout << "selected: " ;
        for (auto s: selected) std::cout << s << " "; std::cout << "\n";
        THROW_LENGTH_ERROR("Lexicase did not select correct number of \
                parents");
    }
    return selected;
}

vector<size_t> FairLexicase::survive(Population& pop, 
        const Parameters& params, const Data& d)
{
    /* FairLexicase survival */
    THROW_RUNTIME_ERROR("Lexicase survival not implemented");
    return vector<size_t>();
}
    
}

}
