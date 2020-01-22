/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "evaluation.h"

// code to evaluate GP programs.
namespace FT{

    using namespace Opt;
    
    namespace Eval{
    
        Evaluation::Evaluation(string scorer)
        {
            score_hash["mse"] = &mse_label;
            score_hash["zero_one"] = &zero_one_loss_label;
            score_hash["bal_zero_one"] = &bal_zero_one_loss_label;
            score_hash["log"] =  &log_loss_label; 
            score_hash["multi_log"] =  &multi_log_loss_label; 
        
            score = score_hash[scorer];
        }

        Evaluation::~Evaluation(){}
                            
        // fitness of population
        void Evaluation::fitness(vector<Individual>& individuals,
                                 const Data& d, 
                                 MatrixXf& F, 
                                 const Parameters& params, 
                                 bool offspring,
                                 bool validation)
        {
        	/*!
             *      @param individuals: population
             *      @param d: Data structure
             *      @param F: matrix of raw fitness values
             *      @param params: algorithm parameters
             *      @param offspring: if true, only evaluate last half of population
             *      @param validation: if true, call ind.predict instead of ind.fit
             
             * Output 
             
             *      F is modified
             *      pop[:].fitness is modified
             */
            
            unsigned start =0;
            if (offspring) start = F.cols()/2;
            
            // loop through individuals
            #pragma omp parallel for
            for (unsigned i = start; i<individuals.size(); ++i)
            {
                Individual& ind = individuals.at(i);

                if (params.backprop)
                {
                    AutoBackProp backprop(params.scorer, params.bp.iters, params.bp.learning_rate);
                    logger.log("Running backprop on " + ind.get_eqn(), 3);
                    backprop.run(ind, d, params);
                }         

                bool pass = true;

                shared_ptr<CLabels> yhat = validation? 
                    ind.predict(d,params) : ind.fit(d,params,pass); 
                // assign F and aggregate fitness
                logger.log("Assigning fitness to " + ind.get_eqn(), 3);

                if (!pass)
                {

                    /* vector<double> w(ind.Phi.rows(), 0);     // set weights to zero */
                    /* ind.set_p(w,params.feedback); */
                    
                    if (validation) 
                        ind.fitness_v = MAX_FLT; 
                    else 
                        ind.fitness = MAX_FLT;

                    F.col(ind.loc) = MAX_FLT*VectorXf::Ones(d.y.size());
                }
                else
                {
                    // assign weights to individual
                    /* ind.set_p(ind.ml->get_weights(),params.feedback); */
                    assign_fit(ind,F,yhat,d.y,params,validation);

                    if (params.hillclimb && !validation)
                    {
                        HillClimb hc(params.scorer, params.hc.iters, params.hc.step);
                        bool updated = false;
                        shared_ptr<CLabels> yhat2 = hc.run(ind, d, params,
                                              updated);
                        if (updated)    // update the fitness of this individual
                        {
                            assign_fit(ind, F, yhat2, d.y, params);
                        }

                    }
                }
            }
        }
        
        // assign fitness to program
        void Evaluation::assign_fit(Individual& ind, MatrixXf& F, const shared_ptr<CLabels>& yhat, 
                                    const VectorXf& y, const Parameters& params, bool val)
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
            VectorXf loss;
            float f = score(y, yhat, loss, params.class_weights);
            
            if (val)
                ind.fitness_v = f;
            else
                ind.fitness = f;
                
            F.col(ind.loc) = loss;  
            
            logger.log("ind " + std::to_string(ind.loc) + " fitness: " + std::to_string(ind.fitness),3);
        }
    }

}
