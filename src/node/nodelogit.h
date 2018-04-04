/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOGIT
#define NODE_LOGIT

#include "node.h"

namespace FT{
	class NodeLogit : public Node
    {
    	public:
    	
    		NodeLogit()
            {
                name = "logit";
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
                stack_f.push_back(1/(1+(limited(exp(-W[0]*x)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("1/(1+exp(-1*" + x + "))");
            }

            ArrayXd getDerivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        numerator = stack_f[stack_f.size() -1] * exp(-W[0] * stack_f[stack_f.size() -1]);
                        denom = pow(1 + np.exp(-W[0] * stack_f[stack_f.size()-1]), 2);
                        return numerator/denom;
                    case 0: // d/dx0
                    default:
                        numerator = W[0] * exp(-W[0] * stack_f[stack_f.size() - 1]);
                        denom = pow(1 + exp(-W[0] * stack_f[stack_f.size() - 1]), 2);
                        return numerator/denom;
                } 
            }

            // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     switch (loc) {
            //         case 0:
            //         default:
            //             numerator = W[0] * exp(-W[0] * stack_f[stack_f.size() - 1]);
            //             denom = pow(1 + exp(-W[0] * stack_f[stack_f.size() - 1]), 2);
            //             gradients.push_back(numerator/denom);
            //     } 
            // }

            // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     update_value = 1
            //     for(auto g : gradients) {
            //         update_value *= g;
            //     }
                 
            //     numerator = stack_f[stack_f.size() -1] * exp(-W[0] * stack_f[stack_f.size() -1]);
            //     denom = pow(1 + np.exp(-W[0] * stack_f[stack_f.size()-1]), 2);
            //     d_w = numerator/denom;
            //     W[0] = W[0] - n/update_value.size * sum(d_w * update_value);
            // }
    };
}	

#endif
