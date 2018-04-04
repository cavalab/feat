/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "node.h"

namespace FT{
	class NodeGaussian : public Node
    {
    	public:
    	
    		NodeGaussian()
            {
                name = "gaussian";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(exp(-1*limited(pow(W[0] * x, 2)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("exp(-(" + x + " ^ 2))");
            }

            ArrayXd getDerivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return -2 * W[0] * pow(stack_f[stack_f.size() - 1], 2) * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                    case 0: // d/dx0
                    default:
                        return -2 * pow(W[0], 2) * stack_f[stack_f.size() - 1] * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                } 
            }
    };
}	

#endif
