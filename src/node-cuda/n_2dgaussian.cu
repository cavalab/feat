/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_2DGAUSSIAN
#define NODE_2DGAUSSIAN

#include "node.h"

namespace FT{
	class Node2dGaussian : public Node
    {
    	public:
    	
    		Node2dGaussian()
            {
                name = "2dgaussian";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x2 = stack_f.back(); stack_f.pop_back();
                ArrayXd x1 = stack_f.back(); stack_f.pop_back();
                
                stack_f.push_back(limited(exp(-1*(pow((x1-x1.mean()), 2)/(2*variance(x1)) 
                                  + pow((x2 - x2.mean()), 2)/variance(x2)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x2 = stack_f.back(); stack_f.pop_back();
                string x1 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("gauss2d(" + x1 + "," + x2 + ")");
            }
            
            private:
                double variance(ArrayXd x)
                {
                    double mean = x.mean();
                    return (limited(pow((x - mean),2))).sum()/(x.count() - 1);
                }
    };
}	

#endif
