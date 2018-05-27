/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "evaluation.h"

// code to evaluate GP programs.
namespace FT{
    
    Evaluation::Evaluation(string scorer)
    {
        score_hash["mse"] = & metrics::mse_label;
        score_hash["zero_one"] = & metrics::zero_one_loss_label;
        score_hash["bal_zero_one"] = & metrics::bal_zero_one_loss_label;
        score_hash["log"] =  & metrics::log_loss_label; 
        score_hash["multi_log"] =  & metrics::multi_log_loss_label; 
    
        score = score_hash[scorer];
    }

    Evaluation::~Evaluation(){}
                        
    // fitness of population
    void Evaluation::fitness(vector<Individual>& individuals,
                             Data d, 
                             MatrixXd& F, 
                             const Parameters& params, 
                             bool offspring)
    {
    	/*!
         * Input:
         
         *      individuals: population
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
        for (unsigned i = start; i<individuals.size(); ++i)
        {
            
            if (params.backprop)
            {
                AutoBackProp backprop(params.scorer, params.bp.iters, params.bp.learning_rate);
                params.msg("Running backprop on " + individuals[i].get_eqn(), 2);
                backprop.run(individuals[i], d, params);
            }            
            // calculate program output matrix Phi
            params.msg("Generating output for " + individuals[i].get_eqn(), 2);
            MatrixXd Phi = individuals.at(i).out(d, params);            

            // calculate ML model from Phi
            params.msg("ML training on " + individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);

            shared_ptr<CLabels> yhat = ml->fit(Phi,d.y,params,pass,individuals[i].dtypes);

            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + individuals[i].get_eqn(), 2);

            if (!pass)
            {
                vector<double> w(0,Phi.rows());     // set weights to zero
                individuals[i].set_p(w,params.feedback);
                individuals[i].fitness = MAX_DBL;
                F.col(individuals[i].loc) = MAX_DBL*VectorXd::Ones(d.y.size());
            }
            else
            {
                // assign weights to individual
                individuals[i].set_p(ml->get_weights(),params.feedback);
                assign_fit(individuals[i],F,yhat,d.y,params);

                if (params.hillclimb)
                {
                    HillClimb hc(params.scorer, params.hc.iters, params.hc.step);
                    bool updated = false;
                    shared_ptr<CLabels> yhat2 = hc.run(individuals.at(i), d, params,
                                          updated);
                    if (updated)    // update the fitness of this individual
                    {
                        assign_fit(individuals[i], F, yhat2, d.y, params);
                    }

                }
            }
        }
    }    
    
    // assign fitness to program
    void Evaluation::assign_fit(Individual& ind, MatrixXd& F, const shared_ptr<CLabels>& yhat, 
                                const VectorXd& y, const Parameters& params, bool val)
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
        double f = score(y, yhat, loss, params.class_weights);
        
        if (val)
            ind.fitness_v = f;
        else
            ind.fitness = f;
        
        F.col(ind.loc) = loss;  
         
        params.msg("ind " + std::to_string(ind.loc) + " fitness: " + std::to_string(ind.fitness),2);
    }

    // validation fitness of population                            
    void Evaluation::val_fitness(vector<Individual>& individuals,
                             Data dt,
                             MatrixXd& F, 
                             Data dv,
                             const Parameters& params, 
                             bool offspring)
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
        for (unsigned i = start; i<individuals.size(); ++i)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + individuals[i].get_eqn(), 2);
            MatrixXd Phi = individuals.at(i).out(dt, params);            

            // calculate ML model from Phi
            params.msg("ML training on " + individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);
            shared_ptr<CLabels> yhat_t = ml->fit(Phi,dt.y,params,pass,individuals[i].dtypes);
            if (!pass)
            {
                HANDLE_ERROR_NO_THROW("Error training eqn " + individuals[i].get_eqn() + 
                                    "\nwith raw output " + 
                                    to_string(individuals[i].out(dt, params)) + "\n");
            }
            
            // calculate program output matrix Phi on validation data
            params.msg("Generating validation output for " + individuals[i].get_eqn(), 2);
            MatrixXd Phi_v = individuals.at(i).out(dv, params);            

            // calculate ML model from Phi
            params.msg("ML predicting on " + individuals[i].get_eqn(), 2);
            shared_ptr<CLabels> yhat_v = ml->predict(Phi_v);

            // assign F and aggregate fitness
            params.msg("Assigning val fitness to " + individuals[i].get_eqn(), 2);
            
            assign_fit(individuals[i],F,yhat_v,dv.y,params, true);
                        
        }
    }
}
