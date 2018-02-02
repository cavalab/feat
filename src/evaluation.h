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
     * @brief evaluation mixin class for Fewtwo
     */
    class Evaluation 
    {
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            
            /// fitness of population.
            void fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                         const Parameters& params, bool offspring);
          
            /// assign fitness to an individual and to F.  
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);       
            
            /* Scoring functions */

            /// 1 - accuracy 
            VectorXd accuracy(const VectorXd& y, const VectorXd& yhat)
            {
                return (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
            }
            
            /// squared error
            VectorXd se(const VectorXd& y, const VectorXd& yhat)
            { 
                return (yhat - y).array().pow(2); 
            };

            /// 1 - balanced accuracy score
            double bal_accuracy(const VectorXd& y, const VectorXd& yhat, unsigned n_classes);
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
            VectorXd yhat = ml->out(Phi,y,params,pass,pop.individuals[i].dtypes);
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
         *       params: fewtwo parameters
         *
         *  Output:
         *
         *       modifies F and ind.fitness
        */ 
        assert(F.cols()>ind.loc);
        if (params.classification)  // use classification accuracy
        {
            F.col(ind.loc) = accuracy(y,yhat); 
            ind.fitness = bal_accuracy(y,yhat,params.n_classes);
        }
        else                        // use mean squared error
        {
            F.col(ind.loc) = se(y, yhat);
            ind.fitness = F.col(ind.loc).mean();
        }
         
        params.msg("ind " + std::to_string(ind.loc) + " fitnes: " + std::to_string(ind.fitness),2);
    }

    double Evaluation::bal_accuracy(const VectorXd& y, const VectorXd& yhat, unsigned n_classes)
    {
         
        // sensitivity (TP) and specificity (TN)
        vector<double> TP(n_classes,0.0), TN(n_classes, 0.0), P(n_classes,0.0), N(n_classes,0.0);
        ArrayXd class_accuracies(n_classes);
        // get class counts
        for (int i=0; i< n_classes; ++i)
        {
            P[i] = (y.array().cast<int>() == i).count();  // total positives for this class
            N[i] = (y.array().cast<int>() != i).count();  // total negatives for this class
        }

        for (unsigned i = 0; i < y.size(); ++i)
        {
            if (yhat(i) == y(i))                    // true positive
                ++TP[y[i]];
            for (unsigned j = 0; j < n_classes; ++i)
                if ( y(i) != j && yhat(i) != j )    // true negative
                    ++TN[j];    
        }
        // class-wise accuracy = 1/2 ( true positive rate + true negative rate)
        for (unsigned i=0; i< n_classes; ++i)
            class_accuracies(i) = (TP[i]/P[i] + TN[i]/N[i])/2; 
        
        return 1.0 - class_accuracies.mean();
    }
}
#endif
