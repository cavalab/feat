/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H
                                                                                                                                      
// internal includes
#include "ml.h"
#include "metrics.h"

using namespace shogun;
using Eigen::Map;

// code to evaluate GP programs.
namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Evaluation
     * @brief evaluation mixin class for Feat
     */
    typedef double (*funcPointer)(const VectorXd&, const shared_ptr<CLabels>&, VectorXd&,
                                  const vector<float>&);
    
    class Evaluation 
    {
        public:
        
            /* VectorXd (* loss_fn)(const VectorXd&, const VectorXd&);  // pointer to loss function */
            double (* score)(const VectorXd&, const shared_ptr<CLabels>&, VectorXd&, 
                             const vector<float>&);    // pointer to scoring function
            std::map<string, funcPointer> score_hash;

            Evaluation(string scorer)
            {
                score_hash["mse"] = & metrics::mse_label;
                score_hash["zero_one"] = & metrics::zero_one_loss_label;
                score_hash["bal_zero_one"] = & metrics::bal_zero_one_loss_label;
                score_hash["log"] =  & metrics::log_loss_label; 
                score_hash["multi_log"] =  & metrics::multi_log_loss_label; 
            
                score = score_hash[scorer];
            }

            ~Evaluation(){}
                
            void set_score(string scorer){ score = score_hash[scorer]; }

            /// fitness of population.
            void fitness(Population& pop,
                         const MatrixXd& X, 
                         const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                         VectorXd& y, 
                         MatrixXd& F, 
                         const Parameters& params, 
                         bool offspring);
          
            void val_fitness(Population& pop,
                             const MatrixXd& X_t,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                             VectorXd& y_t,
                             MatrixXd& F, 
                             const MatrixXd& X_v,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                             VectorXd& y_v,
                             const Parameters& params, 
                             bool offspring);
         
            /// assign fitness to an individual and to F.  
            void assign_fit(Individual& ind, MatrixXd& F, const shared_ptr<CLabels>& yhat, const VectorXd& y,
                            const Parameters& params);       
    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop,
                             const MatrixXd& X,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                             VectorXd& y, 
                             MatrixXd& F, 
                             const Parameters& params, 
                             bool offspring=false)
    {
    	/*!
         * Input:
         
         *      pop: population
         *      X: feature data
         *      y: label
         *      F: matrix of raw fitness values
         *      p: algorithm parameters
         
         * Output:
         
         *      F is modified
         *      pop[:].fitness is modified
         */
        
        
        unsigned start =0;
        if (offspring) start = F.cols()/2;
        // loop through individuals
        #pragma omp parallel for
        for (unsigned i = start; i<pop.size(); ++i)
        {
                        // calculate program output matrix Phi
            params.msg("Generating output for " + pop.individuals[i].get_eqn(), 2);
            MatrixXd Phi = pop.individuals.at(i).out(X, Z, params, y);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);

            shared_ptr<CLabels> yhat = ml->fit(Phi,y,params,pass,pop.individuals[i].dtypes);

            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + pop.individuals[i].get_eqn(), 2);

            if (!pass)
            {
                vector<double> w(0,Phi.rows());     // set weights to zero
                pop.individuals[i].set_p(w,params.feedback);
                pop.individuals[i].fitness = MAX_DBL;
                F.col(pop.individuals[i].loc) = MAX_DBL*VectorXd::Ones(y.size());
            }
            else
            {
                    // assign weights to individual
                    pop.individuals[i].set_p(ml->get_weights(),params.feedback);
                    assign_fit(pop.individuals[i],F,yhat,y,params);

            }

                            
            }
    }    
    
    // assign fitness to program
    void Evaluation::assign_fit(Individual& ind, MatrixXd& F, const shared_ptr<CLabels>& yhat, 
                                const VectorXd& y, const Parameters& params)
    {
        /*!
         * assign raw errors to F, and aggregate fitnesses to individuals. 
         *
         *  Input: 
         *
         *       ind: individual 
         *       F: n_samples x pop_size matrix of errors
         *       yhat: predicted output of ind
         *       y: true labels
         *       params: feat parameters
         *
         *  Output:
         *
         *       modifies F and ind.fitness
        */ 
        assert(F.cols()>ind.loc);
        VectorXd loss;
        ind.fitness = score(y, yhat, loss, params.class_weights);
        F.col(ind.loc) = loss;  
         
        params.msg("ind " + std::to_string(ind.loc) + " fitness: " + std::to_string(ind.fitness),2);
    }

    // validation fitness of population                            
    void Evaluation::val_fitness(Population& pop,
                             const MatrixXd& X_t,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                             VectorXd& y_t,
                             MatrixXd& F, 
                             const MatrixXd& X_v,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                             VectorXd& y_v,
                             const Parameters& params, 
                             bool offspring = false)
    {
    	/*!
         * Input:
         
         *      pop: population
         *      X: feature data
         *      y: label
         *      F: matrix of raw fitness values
         *      p: algorithm parameters
         
         * Output:
         
         *      F is modified
         *      pop[:].fitness is modified
         */
        
        
        unsigned start =0;
        if (offspring) start = F.cols()/2;
        // loop through individuals
        #pragma omp parallel for
        for (unsigned i = start; i<pop.size(); ++i)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + pop.individuals[i].get_eqn(), 2);
            MatrixXd Phi = pop.individuals.at(i).out(X_t, Z_t, params, y_t);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);
            shared_ptr<CLabels> yhat_t = ml->fit(Phi,y_t,params,pass,pop.individuals[i].dtypes);
            if (!pass){
                std::cerr << "Error training eqn " + pop.individuals[i].get_eqn() + "\n";
                std::cerr << "with raw output " << pop.individuals[i].out(X_t, Z_t,params,y_t) << "\n";
                throw;
            }
            
            // calculate program output matrix Phi on validation data
            params.msg("Generating validation output for " + pop.individuals[i].get_eqn(), 2);
            MatrixXd Phi_v = pop.individuals.at(i).out(X_v, Z_v, params, y_v);            

            // calculate ML model from Phi
            params.msg("ML predicting on " + pop.individuals[i].get_eqn(), 2);
            shared_ptr<CLabels> yhat_v = ml->predict(Phi_v);

            // assign F and aggregate fitness
            params.msg("Assigning val fitness to " + pop.individuals[i].get_eqn(), 2);
            
            assign_fit(pop.individuals[i],F,yhat_v,y_v,params);
                        
        }
    }
    
    /* double Evaluation::multi_log_loss(const */ 
    /*         { */
    /*         if (c.empty())  // determine unique class values */
    /*     { */
    /*         vector<double> uc = unique(y); */
    /*         for (const auto& i : uc) */
    /*             c.push_back(int(i)); */
    /*     } */

    /*     //vector<double> class_loss(c.size(),0); */ 
        
    /* } */
}
#endif
