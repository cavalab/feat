/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "node.h"

namespace FT{
	class NodeSign : public Node
    {
    	public:
    	
    		NodeSign()
            {
                name = "sign";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
        		
        		ArrayXd res = (W[0] * x > 0).select(ArrayXd::Ones(x.size()), (x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                stack_f.push_back(res);
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("sign("+ x +")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                // Might want to experiment with using a perceptron update rule or estimating with some other function
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] / (2 * sqrt(W[0] * stack_f[stack_f.size()-1]));
                    case 0: // d/dx0
                    default:
                        return W[0] / (2 * sqrt(W[0] * stack_f[stack_f.size()-1]));
                } 
            }
    };
}	

#endif
