/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "lexicase.h"

namespace FT{


    namespace Sel{

        Lexicase::Lexicase(bool surv){ name = "lexicase"; survival = surv; }
        
        Lexicase::~Lexicase(){}
        
        vector<size_t> Lexicase::select(Population& pop, const MatrixXf& F, 
                const Parameters& params, const Data& d)
        {
            /*! Selection according to lexicase selection for 
             * binary outcomes and epsilon-lexicase selection for continuous. 
             * @param pop: population
             * @param F: n_samples X popsize matrix of model outputs. 
             * @param params: parameters.
             *
             * @return selected: vector of indices corresponding to pop that 
             * are selected.
             *
             * In selection mode, parents are selected among the first half 
             * of columns of F since it is assumed that we are selecting for 
             * offspring to fill the remaining columns. 
             */            

            unsigned int N = F.rows(); //< number of samples
            unsigned int P = F.cols()/2; //< number of individuals
            
            // define epsilon
            ArrayXf epsilon = ArrayXf::Zero(N);
          
            // if output is continuous, use epsilon lexicase            
            if (!params.classification || params.scorer.compare("log")==0 
                    || params.scorer.compare("multi_log")==0)
            {
                // for columns of F, calculate epsilon
                for (int i = 0; i<epsilon.size(); ++i)
                    epsilon(i) = mad(F.row(i));
            }

            // individual locations in F
            vector<size_t> starting_pool;
            for (const auto& p : pop.individuals)
            {
            	starting_pool.push_back(p.loc);
            }
            assert(starting_pool.size() == P);     
            
            vector<size_t> F_locs(P,0); // selected individuals
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
                      if (F(cases[h],pool[j]) < minfit) 
                          minfit = F(cases[h],pool[j]);
                  
                  // select best
                  for (size_t j = 0; j<pool.size(); ++j)
                      if (F(cases[h],pool[j]) <= minfit+epsilon[cases[h]])
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
                F_locs[i] = r.random_choice(winner);   
            }               

            // convert F_locs to pop.individuals indices
            vector<size_t> selected;
            bool match = false;
            for (const auto& f: F_locs)
            {
                for (unsigned i=0; i < pop.size(); ++i)
                {
                    if (pop.individuals[i].loc == f)
                    {
                        selected.push_back(i);
                        match = true;
                    }
                }
                if (!match)
                    HANDLE_ERROR_THROW("no loc matching " + std::to_string(f) 
                            + " in pop");
                match = false;
            }
            if (selected.size() != F_locs.size()){
                std::cout << "pop.locs: ";
                for (auto i: pop.individuals) 
                    std::cout << i.loc << " "; std::cout << "\n";
                std::cout << "selected: " ;
                for (auto s: selected) 
                    std::cout << s << " "; std::cout << "\n";
                std::cout<< "F_locs: ";
                for (auto f: F_locs) 
                    std::cout << f << " "; std::cout << "\n";
            }
            assert(selected.size() == F_locs.size());
            return selected;
        }

        vector<size_t> Lexicase::survive(Population& pop, 
                const MatrixXf& F, const Parameters& params, const Data& d)
        {
            /* Lexicase survival */
        }
        
    }

}
