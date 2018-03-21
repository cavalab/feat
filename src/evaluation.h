/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H
                                                                                                                                      
// internal includes
#include "ml.h"

using namespace shogun;
using Eigen::Map;

// code to evaluate GP programs.
namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Evaluation
     * @brief evaluation mixin class for Feat
     */
    class Evaluation 
    {
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            
            /// fitness of population.
            void fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                         const Parameters& params, bool offspring);
 
            void val_fitness(Population& pop, const MatrixXd& X_t, VectorXd& y_t, MatrixXd& F, 
                             const MatrixXd& X_v, VectorXd& y_v, const Parameters& params, 
                             bool offspring);
         
            /// assign fitness to an individual and to F.  
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);       
            
            /* Scoring functions */

            /// 1 - accuracy 
            VectorXd accuracy(const VectorXd& y, const VectorXd& yhat, bool reverse=true)
            {
                if (reverse)
                    return (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
                return (yhat.cast<int>().array() == y.cast<int>().array()).cast<double>();
            }
            
            /// squared error
            VectorXd se(const VectorXd& y, const VectorXd& yhat)
            { 
                return (yhat - y).array().pow(2); 
            };

            /// 1 - balanced accuracy score
            double bal_accuracy(const VectorXd& y, const VectorXd& yhat, 
                                vector<int> c=vector<int>(), bool reverse=true);
            
            /// log loss
            VectorXd log_loss(const VectorXd& y, const VectorXd& yhat, vector<int> c=vector<int>());
            double bal_loss(const VectorXd& y, const VectorXd& loss, vector<int> c=vector<int>());
    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                 const Parameters& params, bool offspring=false)
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
            MatrixXd Phi = pop.individuals.at(i).out(X, params, y);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);
            VectorXd yhat = ml->fit(Phi,y,params,pass,pop.individuals[i].dtypes);
            if (!pass){
                std::cerr << "Error training eqn " + pop.individuals[i].get_eqn() + "\n";
                std::cerr << "with raw output " << pop.individuals[i].out(X,params,y) << "\n";
                throw;
            }
            // assign weights to individual
           //vector<double> w = ml->get_weights() 
            pop.individuals[i].set_p(ml->get_weights(),params.feedback);
            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + pop.individuals[i].get_eqn(), 2);
            
            assign_fit(pop.individuals[i],F,yhat,y,params);
                        
        }
    }    
    
    // assign fitness to program
    void Evaluation::assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, 
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
        if (params.classification)  // use classification accuracy
        {
            /* F.col(ind.loc) = accuracy(y,yhat); */ 
            F.col(ind.loc) = log_loss(y,yhat); 
            
            /* ind.fitness = bal_accuracy(y,yhat,params.classes); */
            ind.fitness = bal_loss(y,F.col(ind.loc),params.classes);
            
        }
        else                        // use mean squared error
        {
            F.col(ind.loc) = se(y, yhat);
            ind.fitness = F.col(ind.loc).mean();
        }
         
        params.msg("ind " + std::to_string(ind.loc) + " fitnes: " + std::to_string(ind.fitness),2);
    }

    // validation fitness of population
    void Evaluation::val_fitness(Population& pop, const MatrixXd& X_t, VectorXd& y_t, MatrixXd& F, 
                            const MatrixXd& X_v, VectorXd& y_v, const Parameters& params, 
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
            MatrixXd Phi = pop.individuals.at(i).out(X_t, params, y_t);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);
            VectorXd yhat_t = ml->fit(Phi,y_t,params,pass,pop.individuals[i].dtypes);
            if (!pass){
                std::cerr << "Error training eqn " + pop.individuals[i].get_eqn() + "\n";
                std::cerr << "with raw output " << pop.individuals[i].out(X_t,params,y_t) << "\n";
                throw;
            }
            
            // calculate program output matrix Phi on validation data
            params.msg("Generating validation output for " + pop.individuals[i].get_eqn(), 2);
            MatrixXd Phi_v = pop.individuals.at(i).out(X_v, params, y_v);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            VectorXd yhat_v = ml->predict(Phi_v);

            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + pop.individuals[i].get_eqn(), 2);
            
            assign_fit(pop.individuals[i],F,yhat_v,y_v,params);
                        
        }
    }
    double Evaluation::bal_accuracy(const VectorXd& y, const VectorXd& yhat, vector<int> c,
                                    bool reverse)
    {
        if (c.empty())  // determine unique class values
        {
            vector<double> uc = unique(y);
            for (const auto& i : uc)
                c.push_back(int(i));
        }
         
        // sensitivity (TP) and specificity (TN)
        vector<double> TP(c.size(),0.0), TN(c.size(), 0.0), P(c.size(),0.0), N(c.size(),0.0);
        ArrayXd class_accuracies(c.size());
       
        // get class counts
        
        for (unsigned i=0; i< c.size(); ++i)
        {
            P.at(i) = (y.array().cast<int>() == c[i]).count();  // total positives for this class
            N.at(i) = (y.array().cast<int>() != c[i]).count();  // total negatives for this class
        }
        

        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) == y(i))                    // true positive
                ++TP.at(y(i) == -1 ? 0 : y(i));     // if-then ? accounts for -1 class encoding

            for (unsigned j = 0; j < c.size(); ++j)
                if ( y(i) !=c.at(j) && yhat(i) != c.at(j) )    // true negative
                    ++TN.at(j);    
            
        }

        // class-wise accuracy = 1/2 ( true positive rate + true negative rate)
        for (unsigned i=0; i< c.size(); ++i){
            class_accuracies(i) = (TP[i]/P[i] + TN[i]/N[i])/2; 
            //std::cout << "TP(" << i << "): " << TP[i] << ", P[" << i << "]: " << P[i] << "\n";
            //std::cout << "TN(" << i << "): " << TN[i] << ", N[" << i << "]: " << N[i] << "\n";
            //std::cout << "class accuracy(" << i << "): " << class_accuracies(i) << "\n";
        }
        if (reverse)
            return 1.0 - class_accuracies.mean();
        else
            return class_accuracies.mean();
    }

    VectorXd Evaluation::log_loss(const VectorXd& y, const VectorXd& yhat, vector<int> c)

    {
        double eps = pow(10,-10);

        VectorXd loss(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) < eps || 1 - yhat(i) < eps)
                // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
            else
                loss(i) = -(y(i)*log(yhat(i)) + (1-y(i))*log(1-yhat(i)));
        }   
        return loss;
    }

    double Evaluation::bal_loss(const VectorXd& y, const VectorXd& loss, 
                                vector<int> c)
    {
       
        if (c.empty())  // determine unique class values
        {
            vector<double> uc = unique(y);
            for (const auto& i : uc)
                c.push_back(int(i));
        }
        vector<double> class_loss(c.size(),0);

        for (unsigned i = 0; i < c.size(); ++i)
        {
            int n = (y.cast<int>().array() == c[i]).count();
            class_loss[i] = (y.cast<int>().array() == c[i]).select(loss.array(),0).sum()/n;
        
        }
        // return balanced class losses 
        Map<ArrayXd> cl(class_loss.data(),class_loss.size());        
        return cl.mean();
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
