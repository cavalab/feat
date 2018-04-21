/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_TANH
#define NODE_TANH

#include "nodeDx.h"

namespace FT{
	class NodeTanh : public NodeDx
    {
    	public:
    	
    		NodeTanh()
            {
                name = "tanh";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(tanh(W[0] * x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("tanh(" + x + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                ArrayXd numerator;
                ArrayXd denom;
                ArrayXd x = stack_f[stack_f.size()-1];
                switch (loc) {
                    case 1: // d/dw0
                        numerator = 4 * x * exp(2 * this->W[0] * x);
                        denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                        // numerator = 4 * x * exp(2 * this->W[0] * x - 1]); 
                        // denom = pow(exp(2 * this->W[0] * x) + 1,2);
                        return numerator/denom;
                    case 0: // d/dx0
                    default:
                        numerator = 4 * this->W[0] * exp(2 * this->W[0] * x);
                        denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                        // numerator = 4 * W[0] * exp(2 * W[0] * stack_f[stack_f.size() - 1]);
                        // denom = pow(exp(2 * W[0] * stack_f[stack_f.size()-1]),2);
                        return numerator/denom;
                } 
            }

            // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     switch (loc) {
            //         case 0:
            //         default:
            //             numerator = 4 * W[0] * exp(2 * W[0] * stack_f[stack_f.size() - 1]);
            //             denom = pow(exp(2 * W[0] * stack_f[stack_f.size() - 1]),2);
            //             gradients.push_back(numerator/denom);
            //             break;
            //     } 
            // }

            // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, double n) {
            //     int update_value = 1;
            //     for(auto &grad : gradients) {
            //         update_value *= grad;
            //     }

            //     numerator = 4 * stack_f[stack_f.size() - 1] * exp(2 * W[0] * stack_f[stack_f.size() - 1]); 
            //     denom = pow(exp(2 * W[0] * stack_f[stack_f.size()-1]) + 1,2);
            //     d_w = numerator/denom;
            //     W[0] = W[0] - n/update_value.size() * sum(d_w * update_value);
            // }
        protected:
            NodeTanh* clone_impl() const override { return new NodeTanh(*this); };  
    };
}	

#endif
