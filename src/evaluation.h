/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H

// code to evaluate GP programs.
namespace FT{
    class Evaluation 
    {
        // evaluation mixin class for Fewtwo
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            // fitness of population.
            void fitness(Population& pop, const MatrixXd& X, const VectorXd& y, const MatrixXd& F, 
                         const Parameters& p);
        private:
            
            

    };
}
#endif
