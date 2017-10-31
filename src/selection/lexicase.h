/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef LEXICASE_H
#define LEXICASE_H

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Lexicase
     * @brief Lexicase selection operator.
     */
    struct Lexicase : SelectionOperator
    {
        Lexicase(bool surv){ name = "lexicase"; survival = surv; }
        
        ~Lexicase(){}

        /// function returns a set of selected indices from F. 
        vector<size_t> select(const MatrixXd& F, const Parameters& params, Rnd& r); 
        
        /// lexicase survival
        vector<size_t> survive(const MatrixXd& F, const Parameters& params, Rnd& r); 

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    vector<size_t> Lexicase::select(const MatrixXd& F, const Parameters& params, Rnd& r)
    {
        /* Selection according to lexicase selection for classification and epsilon-lexicase
         * selection for regression. 
         *
         * Input: 
         *
         *      F: n_samples X popsize matrix of model outputs. 
         *      params: parameters.
         *
         * Output:
         *
         *      selected: vector of indices corresponding to columns of F that are selected.
         *      In selection mode, parents are selected among the first half of columns of F since
         *      it is assumed that we are selecting for offspring to fill the remaining columns. 
         */            
        
        if (survival)               // if survival is on, run lexicase survival.
            return survive(F,params,r);

        unsigned int N = F.rows(); //< number of samples
        unsigned int P = F.cols()/2; //< number of individuals

        // define epsilon
        ArrayXd epsilon = ArrayXd::Zero(N);
      
        // if output is continuous, use epsilon lexicase            
        if (!params.classification)
        {
            // for columns of F, calculate epsilon
            for (int i = 0; i<epsilon.size(); ++i)
                epsilon(i) = mad(F.row(i));
        }

        // individual locations
        vector<size_t> starting_pool(P);                    
        iota(starting_pool.begin(), starting_pool.end(), 0);
        
        
        vector<size_t> selected(P,0); // selected individuals
        
        #pragma omp parallel for 
        for (unsigned int i = 0; i<P; ++i)  // selection loop
        {
 
            vector<size_t> cases(N);                // cases
            iota(cases.begin(),cases.end(),0);
            vector<size_t> pool = starting_pool;    // initial pool   
            vector<size_t> winner;                  // winners
            r.shuffle(cases.begin(),cases.end());   // shuffle cases
    
            bool pass = true;     // checks pool size and number of cases
            unsigned int h = 0;   // case count

            while(pass){    // main loop

              winner.resize(0);   // winners                  
              double minfit = std::numeric_limits<double>::max();   // minimum error on case
            
              for (size_t j = 0; j<pool.size(); ++j)
              {
                  if (F(cases[h],pool[j]) < minfit+epsilon[cases[h]])
                  {                          
                      minfit=F(cases[h],pool[j]);     // set the new minimum
                      winner.resize(0);               // start the winners pool
                      winner.push_back(pool[j]);     
                  }
                  else if (F(cases[h],pool[j]) == minfit+epsilon[cases[h]])
                      winner.push_back(pool[j]);	  
              }
             
              ++h; // next case
              pass = (winner.size()>1 && h<cases.size()); // only keep going if needed
              pool = winner;    // reduce pool to remaining individuals
          
            }       

            assert(winner.size()>0);
            //if more than one winner, pick randomly
            selected[i] = r.random_choice(winner);                            
        }               

        return selected;
    }

    vector<size_t> Lexicase::survive(const MatrixXd& F, const Parameters& params, Rnd& r)
    {
        /* Lexicase survival */
    }

}
#endif
