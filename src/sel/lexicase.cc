/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "lexicase.h"

namespace FT{
namespace Sel{

Lexicase::Lexicase(bool surv){ name = "lexicase"; survival = surv; }

Lexicase::~Lexicase(){}

vector<size_t> Lexicase::select(Population& pop,  
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
    // define epsilon
    ArrayXf epsilon = ArrayXf::Zero(N);
  
    // if output is continuous, use epsilon lexicase            
    if (!params.classification || params.scorer_.compare("log")==0 
            || params.scorer_.compare("multi_log")==0)
    {
        // for each sample, calculate epsilon
        for (int i = 0; i<epsilon.size(); ++i)
        {
            VectorXf case_errors(pop.individuals.size());
            for (int j = 0; j<pop.individuals.size(); ++j)
            {
                case_errors(j) = pop.individuals.at(j).error(i);
            }
            epsilon(i) = mad(case_errors);
        }
    }

    // selection pool
    vector<size_t> starting_pool;
    for (int i = 0; i < pop.individuals.size(); ++i)
    {
        starting_pool.push_back(i);
    }
    assert(starting_pool.size() == P);     
    
    vector<size_t> selected(P,0); // selected individuals
    #pragma omp parallel for 
    for (unsigned int i = 0; i<P; ++i)  // selection loop
    {
        vector<size_t> cases; // cases (samples)
        if (params.classification && !params.class_weights.empty()) 
        {
            // for classification problems, weight case selection 
            // by class weights
            vector<size_t> choices(N);
            std::iota(choices.begin(), choices.end(),0);
            vector<float> sample_weights = params.sample_weights;
            for (unsigned i = 0; i<N; ++i)
            {
                vector<size_t> choice_idxs(N-i);
                std::iota(choice_idxs.begin(),choice_idxs.end(),0);
                size_t idx = r.random_choice(choice_idxs,
                        sample_weights);
                cases.push_back(choices.at(idx));
                choices.erase(choices.begin() + idx);
                sample_weights.erase(sample_weights.begin() + idx);
            }
        }
        else
        {   // otherwise, choose cases randomly
            cases.resize(N); 
            std::iota(cases.begin(),cases.end(),0);
            r.shuffle(cases.begin(),cases.end());   // shuffle cases
        }
        vector<size_t> pool = starting_pool;    // initial pool   
        vector<size_t> winner;                  // winners

        bool pass = true;     // checks pool size and number of cases
        unsigned int h = 0;   // case count

        while(pass){    // main loop

          winner.resize(0);   // winners                  
          // minimum error on case
          float minfit = std::numeric_limits<float>::max();                     

          // get minimum
          for (size_t j = 0; j<pool.size(); ++j)
              if (pop.individuals.at(pool[j]).error(cases[h]) < minfit) 
                  minfit = pop.individuals.at(pool[j]).error(cases[h]);
          
          // select best
          for (size_t j = 0; j<pool.size(); ++j)
              if (pop.individuals.at(pool[j]).error(cases[h]) 
                      <= minfit+epsilon[cases[h]])
                winner.push_back(pool[j]);                 
         
          ++h; // next case
          // only keep going if needed
          pass = (winner.size()>1 && h<cases.size()); 
          
          if(winner.size() == 0)
          {
            if(h >= cases.size())
                winner.push_back(r.random_choice(pool));
            else
                pass = true;
          }
          else
            pool = winner;    // reduce pool to remaining individuals
      
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

vector<size_t> Lexicase::survive(Population& pop, 
        const Parameters& params, const Data& d)
{
    /* Lexicase survival */
    THROW_RUNTIME_ERROR("Lexicase survival not implemented");
}

}

}
