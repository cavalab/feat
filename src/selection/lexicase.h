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
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params); 
        
        /// lexicase survival
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params); 

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    vector<size_t> Lexicase::select(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /*! Selection according to lexicase selection for classification and epsilon-lexicase
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

        // individual locations in F
        vector<size_t> starting_pool;
        for (const auto& p : pop.individuals) starting_pool.push_back(p.loc);
        assert(starting_pool.size() == P);     
        
        vector<size_t> F_locs(P,0); // selected individuals
        
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
              
              // get minimum
              for (size_t j = 0; j<pool.size(); ++j)
                  if (F(cases[h],pool[j]) < minfit) 
                      minfit = F(cases[h],pool[j]);
              
              // select best
              for (size_t j = 0; j<pool.size(); ++j)
                  if (F(cases[h],pool[j]) <= minfit+epsilon[cases[h]])
                    winner.push_back(pool[j]);                 
             
              ++h; // next case
              pass = (winner.size()>1 && h<cases.size()); // only keep going if needed
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
                std::cout << "no loc matching " << f << " in pop\n";
            match = false;
        }
        if (selected.size() != F_locs.size()){
            std::cout << "pop.locs: ";
            for (auto i: pop.individuals) std::cout << i.loc << " "; std::cout << "\n";
            std::cout << "selected: " ;
            for (auto s: selected) std::cout << s << " "; std::cout << "\n";
            std::cout<< "F_locs: ";
            for (auto f: F_locs) std::cout << f << " "; std::cout << "\n";
        }
        assert(selected.size() == F_locs.size());
        return selected;
    }

    vector<size_t> Lexicase::survive(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Lexicase survival */
    }

}
#endif
