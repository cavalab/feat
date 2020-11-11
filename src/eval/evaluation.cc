/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "evaluation.h"

// code to evaluate GP programs.
namespace FT{

    using namespace Opt;
    
    namespace Eval{
    
        Evaluation::Evaluation(string scorer): S(scorer)
        {
            this->S.set_scorer(scorer);
        }

        Evaluation::~Evaluation(){}
                            
        void Evaluation::validation(vector<Individual>& individuals,
                                 const Data& d, 
                                 const Parameters& params, 
                                 bool offspring
                )
        {
            unsigned start =0;
            if (offspring) 
                start = individuals.size()/2;

            // loop through individuals
            /* #pragma omp parallel for */
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
                        + ", id: " + to_string(ind.id), 3);

                shared_ptr<CLabels> yhat =  ind.predict(d);
                // assign aggregate fitness
                logger.log("Assigning fitness to ind " + to_string(i) 
                        + ", eqn: " + ind.get_eqn(), 3);

                if (!pass)
                {

                    ind.fitness_v = MAX_FLT; 
                }
                else
                {
                    // assign fitness to individual
                    VectorXf loss;
                    ind.fitness_v = this->S.score(d.y, yhat, loss, 
                                            params.class_weights);
                }
            }
        }
        // fitness of population
        void Evaluation::fitness(vector<Individual>& individuals,
                                 const Data& d, 
                                 const Parameters& params, 
                                 bool offspring)
        {
        	/*!
             *      @param individuals: population
             *      @param d: Data structure
             *      @param params: algorithm parameters
             *      @param offspring: if true, only evaluate last half of population
             
             * Output 
             
             *      individuals.fitness, yhat, error is modified
             */
            
            unsigned start =0;
            if (offspring) start = individuals.size()/2;

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
                    #pragma omp critical
                    {
                    AutoBackProp backprop(params.scorer_, params.bp.iters, 
                            params.bp.learning_rate);
                    logger.log("Running backprop on " + ind.get_eqn(), 3);
                    backprop.run(ind, d, params);
                    }         
                }
                bool pass = true;

                logger.log("Running ind " + to_string(i) 
                        + ", id: " + to_string(ind.id), 3);

                shared_ptr<CLabels> yhat =  ind.fit(d,params,pass); 
                // assign F and aggregate fitness
                logger.log("Assigning fitness to ind " + to_string(i) 
                        + ", eqn: " + ind.get_eqn(), 3);

                if (!pass)
                {

                    ind.fitness = MAX_FLT;
                    ind.error = MAX_FLT*VectorXf::Ones(d.y.size());
                }
                else
                {
                    // assign weights to individual
                    assign_fit(ind,yhat,d,params,false);
                    

                    if (params.hillclimb)
                    {
                        HillClimb hc(params.scorer_, params.hc.iters, 
                                params.hc.step);
                        bool updated = false;
                        shared_ptr<CLabels> yhat2 = hc.run(ind, d, params,
                                              updated);
                        // update the fitness of this individual
                        if (updated)    
                        {
                            assign_fit(ind, yhat2, d, params);
                        }

                    }
                }
            }
        }
        
        // assign fitness to program
        void Evaluation::assign_fit(Individual& ind,  
                const shared_ptr<CLabels>& yhat, const Data& d, 
                const Parameters& params, bool val)
        {
            /*!
             * assign raw errors and aggregate fitnesses to individuals. 
             *
             *  Input: 
             *
             *       ind: individual 
             *       yhat: predicted output of ind
             *       d: data
             *       params: feat parameters
             *
             *  Output:
             *
             *       modifies individual metrics
            */ 
            VectorXf loss;
            float f = S.score(d.y, yhat, loss, params.class_weights);
            //TODO: add if condition for this
            float fairness = marginal_fairness(loss, d, f);
            
            if (fairness <0 )
            {
                cout << "fairness is " << fairness << "...\n";
            }
            if (val)
            {
                ind.fitness_v = f;
                ind.fairness_v = fairness;
            }
            else
            {
                ind.fitness = f;
                ind.fairness = fairness;
                ind.error = loss;
            }
                
            logger.log("ind " + std::to_string(ind.id) + " fitness: " 
                    + std::to_string(ind.fitness),3);
        }

        float Evaluation::marginal_fairness(VectorXf& loss, const Data& d, 
                float base_score, bool use_alpha)
        {
            // averages the deviation of the loss function from average loss 
            // over k
            float avg_score = 0;
            float count = 0;
            float alpha = 1;

            ArrayXb x_idx; 

            for (const auto& pl : d.protect_levels)
            {
                for (const auto& lvl : pl.second)
                {
                    x_idx = (d.X.row(pl.first).array() == lvl);
                    float len_g = x_idx.count();
                    if (use_alpha)
                        alpha = len_g/d.X.cols();
                    /* cout << "alpha = " << len_g << "/" 
                     * << d.X.cols() << endl; */
                    float Beta = fabs(base_score - 
                                    x_idx.select(loss,0).sum()/len_g);
                    /* cout << "Beta = |" << base_score << " - " */ 
                    /*     << x_idx.select(loss,0).sum() << "/" */ 
                    /*     << len_g << "|" << endl; */
                    avg_score += alpha * Beta;
                    ++count;
                }

            }
            avg_score /= count;
            if (std::isinf(avg_score) 
                    || std::isnan(avg_score)
                    || avg_score < 0)
                return MAX_FLT;
                
            return avg_score; 

        }
    }
}
