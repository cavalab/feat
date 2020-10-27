/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "../dat/state.h"
#ifdef USE_CUDA
    #include "cuda-op/cuda_utils.h"
#endif
#include "../dat/data.h"
#include "../params.h"
#include "../model/ml.h"
#include "../util/utils.h"
#include "../util/serialization.h"
#include "nodevector.h"

namespace FT{

using namespace Model;

namespace Pop{
    
////////////////////////////////////////////////////////// Declarations

/*!
 * @class Individual
 * @brief individual programs in the population
 */
class Individual{
public:        
    NodeVector program; ///< executable data structure
    MatrixXf Phi;      ///< transformation output of program 
    VectorXf yhat;     ///< current output
    VectorXf error;     ///< training error
    shared_ptr<ML> ml; ///< ML model, trained on Phi
    float fitness;     ///< aggregate fitness score
    float fitness_v;   ///< aggregate validation fitness score
    float fairness;     ///< aggregate fairness score
    float fairness_v;   ///< aggregate validation fairness score
    vector<float> w;   ///< weights from ML training on program output
    vector<float> p;   ///< probability of variation of subprograms
    unsigned int dim;  ///< dimensionality of individual
    vector<float> obj; ///< objectives for use with Pareto selection
    unsigned int dcounter;  ///< number of individuals this dominates
    vector<unsigned int> dominated; ///< individual indices this dominates
    unsigned int rank;             ///< pareto front rank
    float crowd_dist;   ///< crowding distance on the Pareto front
    unsigned int complexity; ///< the complexity of the program.    
    vector<char> dtypes;    ///< the data types of each column of the 
                                                  // program output
    unsigned id;                                ///< tracking id
    vector<int> parent_id;                      ///< ids of parents
    string eqn; ///< equation form of the program
   
    Individual();

    /// copy assignment
    /* Individual(const Individual& other); */
    
    /* Individual(Individual && other); */
    
    /* Individual& operator=(Individual const& other); */
    
    /* Individual& operator=(Individual && other); */
    void initialize(const Parameters& params, bool random, int id=0);

    /// calculate program output matrix Phi
    MatrixXf out(const Data& d, bool predict=false);

    /// calculate program output while maintaining stack trace
    MatrixXf out_trace(const Data& d, vector<Trace>& stack_trace);

    /// converts program states to output matrices
    MatrixXf state_to_phi(State& state);

    /// fits an ML model to the data after transformation
    shared_ptr<CLabels> fit(const Data& d, const Parameters& params, 
            bool& pass);
    /// fits an ML model to the data after transformation
    shared_ptr<CLabels> fit(const Data& d, const Parameters& params);
    
    /// fits and tunes an ML model to the data after transformation
    shared_ptr<CLabels> fit_tune(const Data& d, 
            const Parameters& params, bool set_default=false);

    /// tunes an ML model to the data after transformation
    void tune(const Data& d, const Parameters& params);
    /*! generates prediction on data using transformation and ML predict. 
     *  @param drop_idx if specified, the phi output at drop_idx is set to zero, effectively
     *  removing its output from the transformation. used in semantic crossover.
     */
    shared_ptr<CLabels> predict(const Data& d);
    VectorXf predict_vector(const Data& d);
    ArrayXXf predict_proba(const Data& d);
    /// return symbolic representation of program
    string get_eqn() ;

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
    
    /// calculate program complexity and return it. 
    unsigned int set_complexity();
   
    /// get the program complexity without updating it.
    unsigned int get_complexity() const;
  
    /// clone this individual 
    void clone(Individual& cpy, bool sameid=true) const;
    Individual clone();
    
    void set_id(unsigned i);

    /// set parent ids using parents  
    void set_parents(const vector<Individual>& parents);
    
    /// set parent ids using id values 
    void set_parents(const vector<int>& parents){ parent_id = parents; }

    /// get probabilities of variation
    vector<float> get_p() const;
    
    /// get inverted weight probability for pogram location i
    float get_p(const size_t i, bool normalize=true) const;
    
    /// get probability of variation for program locations locs
    vector<float> get_p(const vector<size_t>& locs, bool normalize=false) const; 

    /// set probabilities
    void set_p(const vector<float>& weights, const float& fb, 
               const bool softmax_norm=false);
    
    /// get maximum stack size needed for evaluation.
    std::map<char,size_t> get_max_state_size();
    
    /// save individual as a json object.
    void save(string filename);
    /// load individual from a file. 
    void load(string filename);

    typedef Array<bool, Dynamic, Dynamic, RowMajor> ArrayXXb;
    /* typedef Array<float, Dynamic, Dynamic, RowMajor> ArrayXXf; */

};

// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Individual, 
        program,
        /* Phi, */
        /* yhat, */
        /* error, */
        eqn,
        ml,
        fitness,
        fitness_v,
        fairness,
        fairness_v,
        w,
        p,
        dim,
        obj,
        dcounter,
        dominated,
        rank,
        crowd_dist,
        complexity,
        dtypes,
        id,
        parent_id
        )
}
}

#endif
