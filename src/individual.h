/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "stack.h"
#include "data.h"
#include "params.h"
#include "ml.h"
#include "utils.h"

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class Individual
     * @brief individual programs in the population
     */
    class Individual{
    public:        
        NodeVector program;                         ///< executable data structure
        MatrixXd Phi;                               ///< transformation output of program 
        VectorXd yhat;                              ///< current output
        shared_ptr<ML> ml;                          ///< ML model, trained on Phi
        double fitness;             				///< aggregate fitness score
        double fitness_v;             				///< aggregate validation fitness score
        size_t loc;                 				///< index of individual in semantic matrix F
        string eqn;                 				///< symbolic representation of program
        vector<double> w;            				///< weights from ML training on program output
        vector<double> p;                           ///< probability of variation of subprograms
        unsigned int dim;           				///< dimensionality of individual
        vector<double> obj;                         ///< objectives for use with Pareto selection
        unsigned int dcounter;                      ///< number of individuals this dominates
        vector<unsigned int> dominated;             ///< individual indices this dominates
        unsigned int rank;                          ///< pareto front rank
        float crowd_dist;                           ///< crowding distance on the Pareto front
        unsigned int c;                             ///< the complexity of the program.    
        vector<char> dtypes;                        ///< the data types of each column of the 
                                                      // program output
        unsigned id;                                ///< tracking id
        vector<unsigned> parent_id;                 ///< ids of parents
       
        Individual();

        /// calculate program output matrix Phi
        MatrixXd out(const Data& d, const Parameters& params, bool predict=false);

        /// calculate program output while maintaining stack trace
        MatrixXd out_trace(const Data& d,
                     const Parameters& params, vector<Trace>& stack_trace);

        /// fits an ML model to the data after transformation
        shared_ptr<CLabels> fit(const Data& d, const Parameters& params, bool& pass);
        
        /*! generates prediction on data using transformation and ML predict. 
         *  @params drop_idx if specified, the phi output at drop_idx is set to zero, effectively
         *  removing its output from the transformation. used in semantic crossover.
         */
        shared_ptr<CLabels> predict(const Data& d, const Parameters& params);
        VectorXd predict_vector(const Data& d, const Parameters& params);
        VectorXd predict_drop(const Data& d, const Parameters& params, int drop_idx);
        /// return symbolic representation of program
        string get_eqn();

        /// return vectorized representation of program
        vector<string> get_features();

        /// return program name list 
        string program_str() const;

        /// setting and getting from individuals vector
        /* const std::unique_ptr<Node> operator [](int i) const {return program.at(i);} */ 
        /* const std::unique_ptr<Node> & operator [](int i) {return program.at(i);} */

        /// set rank
        void set_rank(unsigned r);
        
        /// return size of program
        int size() const;
        
        /// get number of params in program
        int get_n_params();
        
        /// grab sub-tree locations given starting point.
        /* size_t subtree(size_t i, char otype) const; */

        // // get program depth.
        // unsigned int depth();

        /// get program dimensionality
        unsigned int get_dim();
        
        /// check whether this dominates b. 
        int check_dominance(const Individual& b) const;
        
        /// set obj vector given a string of objective names
        void set_obj(const vector<string>&); 
        
        /// calculate program complexity. 
        unsigned int complexity();
        
        unsigned int get_complexity() const;
      
        /// clone this individual 
        void clone(Individual& cpy, bool sameid=true);
        
        void set_id(unsigned i);
        
        void set_parents(const vector<Individual>& parents);
        
        /// get probabilities of variation
        vector<double> get_p() const;
        
        /// get inverted weight probability for pogram location i
        double get_p(const size_t i) const;
        
        /// get probability of variation for program locations locs
        vector<double> get_p(const vector<size_t>& locs) const; 

        /// set probabilities
        void set_p(const vector<double>& weights, const double& fb);
    };
}

#endif
