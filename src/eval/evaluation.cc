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
        
            score = score_hash.at(scorer);
        }

        Evaluation::~Evaluation(){}
                            
        void Evaluation::validation(vector<Individual>& individuals,
                                 const Data& d, 
                                 MatrixXf& F, 
                                 const Parameters& params, 
                                 bool offspring
                )
        {
            unsigned start =0;
            if (offspring) start = F.cols()/2;

            // loop through individuals
            #pragma omp parallel for
            for (unsigned i = start; i<individuals.size(); ++i)
            {
                Individual& ind = individuals.at(i);
                
                // if there is no validation data,
                // set fitness_v to fitness and return
                if (d.X.cols() == 0) 
                {
                    ind.fitness_v = ind.fitness;
                    continue;
                }

                bool pass = true;

                logger.log("Validating ind " + to_string(i) 
                        + ", location: " + to_string(ind.loc), 3);

                shared_ptr<CLabels> yhat =  ind.predict(d,params);
                // assign F and aggregate fitness
                logger.log("Assigning fitness to ind " + to_string(i) 
                        + ", location: " + to_string(ind.loc) 
                        + ", eqn: " + ind.get_eqn(), 3);

                if (!pass)
                {

                    ind.fitness_v = MAX_FLT; 
                    F.col(ind.loc) = MAX_FLT*VectorXf::Ones(d.y.size());
                }
                else
                {
                    // assign fitness to individual
                    assign_fit(ind,F,yhat,d.y,params,true);
                }
            }
        }
        // fitness of population
        void Evaluation::fitness(vector<Individual>& individuals,
                                 const Data& d, 
                                 MatrixXf& F, 
                                 const Parameters& params, 
                                 bool offspring)
        {
        	/*!
             *      @param individuals: population
             *      @param d: Data structure
             *      @param F: matrix of raw fitness values
             *      @param params: algorithm parameters
             *      @param offspring: if true, only evaluate last half of population
             
             * Output 
             
             *      F is modified
             *      pop[:].fitness is modified
             */
            
            unsigned start =0;
            if (offspring) start = F.cols()/2;

            /* for (unsigned i = start; i<individuals.size(); ++i) */
            /* { */
            /*     cout << "ind " << i << " size: " */ 
            /*         << individuals.at(i).size() << endl; */
            /*     /1* cout << "ind " << i << " eqn: " *1/ */ 
            /*     /1*     << individuals.at(i).get_eqn() << endl; *1/ */
            /*     /1* cout << "ind " << i << " program str: " *1/ */ 
            /*     /1*     << individuals.at(i).program_str() << endl; *1/ */
            /* } */
            
            // loop through individuals
            #pragma omp parallel for
            for (unsigned i = start; i<individuals.size(); ++i)
            {
                Individual& ind = individuals.at(i);

                if (params.backprop)
                {
                    AutoBackProp backprop(params.scorer, params.bp.iters, 
                            params.bp.learning_rate);
                    logger.log("Running backprop on " + ind.get_eqn(), 3);
                    backprop.run(ind, d, params);
                }         

                bool pass = true;

                logger.log("Running ind " + to_string(i) 
                        + ", location: " + to_string(ind.loc), 3);

                shared_ptr<CLabels> yhat =  ind.fit(d,params,pass); 
                // assign F and aggregate fitness
                logger.log("Assigning fitness to ind " + to_string(i) 
                        + ", location: " + to_string(ind.loc) 
                        + ", eqn: " + ind.get_eqn(), 3);

                if (!pass)
                {

                    ind.fitness = MAX_FLT;
                    F.col(ind.loc) = MAX_FLT*VectorXf::Ones(d.y.size());
                }
                else
                {
                    // assign fitness to individual
                    assign_fit(ind,F,yhat,d.y,params,false);

                    if (params.hillclimb)
                    {
                        HillClimb hc(params.scorer, params.hc.iters, 
                                params.hc.step);
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
        void Evaluation::assign_fit(Individual& ind, MatrixXf& F, 
                const shared_ptr<CLabels>& yhat, const VectorXf& y, 
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
            
            logger.log("ind " + std::to_string(ind.loc) + " fitness: " 
                    + std::to_string(ind.fitness),3);
        }
    }
}
