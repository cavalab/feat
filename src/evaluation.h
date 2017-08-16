/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H

// code to evaluate GP programs.
namespace FT{
    
    //////////////////////////////////////////////////////////////////////////////// Declarations
    
    class Evaluation 
    {
        // evaluation mixin class for Fewtwo
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            // fitness of population.
            void fitness(Population& pop, const MatrixXd& X, const VectorXd& y, MatrixXd& F, 
                         const Parameters& params);

            // output of an ml model. 
            VectorXd out_ml(const MatrixXd& Phi, const VectorXd& y, const Parameters& params);

            // assign fitness to an individual and to F. 
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);
        private:
            
            

    };
    
    /////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop, const MatrixXd& X, const VectorXd& y, MatrixXd& F, 
                 const Parameters& p)
    {
        // Input:
        //      pop: population
        //      X: feature data
        //      y: label
        //      F: matrix of raw fitness values
        //      p: algorithm parameters
        // Output:
        //      F is modified
        //      pop[:].fitness is modified
        
        
        // loop through individuals
        for (auto ind : pop.individuals)
        {
            // calculate program output matrix Phi
            MatrixXd Phi = ind.out(X, y, p);
            

            // calculate ML model from Phi
            VectorXd yhat = out_ml(Phi,y,p);
            
            // assign F and aggregate fitness
            assign_fit(ind,F,yhat,y,p);
            
            
        }

     
    }
    

    // train ML model and generate output
    VectorXd Evaluation::out_ml(const MatrixXd& Phi, const VectorXd& y,const Parameters& params)
    {
    }
    
    // assign fitness to program
    void Evaluation::assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, 
                                const VectorXd& y, const Parameters& params)
    {
    }
}
#endif
