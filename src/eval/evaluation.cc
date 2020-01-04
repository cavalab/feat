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
            score_hash["fpr"] =  &false_positive_loss_label; 
        
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
                    AutoBackProp backprop(params.scorer, 
                            params.bp.iters, params.bp.learning_rate);
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
                    assign_fit(ind,F,yhat,d,params,validation);

                    if (params.hillclimb && !validation)
                    {
                        HillClimb hc(params.scorer, params.hc.iters, 
                                params.hc.step);
                        bool updated = false;
                        shared_ptr<CLabels> yhat2 = hc.run(ind, d, params,
                                              updated);
                        // update the fitness of this individual
                        if (updated)    
                        {
                            assign_fit(ind, F, yhat2, d, params);
                        }

                    }
                }
            }
        }
        
        // assign fitness to program
        void Evaluation::assign_fit(Individual& ind, MatrixXf& F, 
                const shared_ptr<CLabels>& yhat, const Data& d, 
                const Parameters& params, bool val)
        {
            /*!
             * assign raw errors to F, and aggregate fitnesses to individuals. 
             *
             *  Input: 
             *
             *       ind: individual 
             *       F: n_samples x pop_size matrix of errors
             *       yhat: predicted output of ind
             *       d: data
             *       params: feat parameters
             *
             *  Output:
             *
             *       modifies F and ind.fitness
            */ 
            assert(F.cols()>ind.loc);
            VectorXf loss;
            float f = score(d.y, yhat, loss, params.class_weights);
            float fairness = subgroup_fairness(loss, d, f);
            
            if (val)
            {
                ind.fitness_v = f;
                ind.fairness_v = fairness;
            }
            else
            {
                ind.fitness = f;
                ind.fairness = fairness;
            }
                
            F.col(ind.loc) = loss;  
            
            logger.log("ind " + std::to_string(ind.loc) + " fitness: " 
                    + std::to_string(ind.fitness),3);
        }

        float Evaluation::subgroup_fairness(VectorXf& loss, const Data& d, 
                float base_score)
        {
            // averages the deviation of the loss function from average over
            // k
            float avg_score = 0;
            float count = 0;

            ArrayXb x_idx = ArrayXb::Constant(d.X.cols(),true);

            for (const auto& pl : d.protect_levels)
            {
                for (const auto& lvl : pl.second)
                {
                    x_idx = (d.X.row(pl.first).array() == lvl);
                    float len_g = x_idx.count();
                    float alpha = len_g/d.X.rows();
                    float Beta = fabs(base_score - 
                                    x_idx.select(loss,0).sum()/len_g);
                    avg_score += alpha * Beta;
                    ++count;
                }

            }
            return avg_score / count; 

        }
    }

}
