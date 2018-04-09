/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_RELU
#define NODE_RELU

#include "nodeDx.h"

namespace FT{
	class NodeDivide : public Node
    {
    	public:
    	  	
    		NodeDivide()
    		{
    			name = "/";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
                ArrayXd x = stack_f.back(); stack_f.pop_back();
                
                ArrayXd res = (W[0] * x > 0).select(, (W[0] * x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                stack_f.push_back(res);
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("relu("+ x +")");         	
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dx1
                        ArrayXd x = fwd_stack[fwd_stack.size()-1];
                        ArrayXd res = (W[0] * x > 0).select(ArrayXd::Ones(x.size()), (x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                        return W[1]/(W[0] * fwd_stack[-2]);
                    case 0: // d/dx0
                    default:
                        ArrayXd x = fwd_stack[fwd_stack.size()-1];
                        ArrayXd res = (W[0] * x > 0).select(ArrayXd::Ones(x.size()), (x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                        return -W[1] * stack_f[stack_f.size() - 1]/(W[0] * pow(stack_f[stack_f.size()], 2));
                } 
            }

            // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     switch (loc) {
            //         case 1:
            //             gradients.push_back(-W[0] * stack_f[stack_f.size() - 1]/
            //                 (W[1] * pow(stack_f[stack_f.size()], 2)));
            //             break;
            //         case 0:
            //         default:
            //             gradients.push_back(W[0]/(W[1] * fwd_stack[-2]));
            //     } 
            // }

            // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     update_value = 1
            //     for(auto g : gradients) {
            //         update_value *= g;
            //     }
                 
            //     W_temp = W[:]
            //     d_w = stack_f[stack_f.size()-1]/(W_temp[1] * stack_f[stack_f.size()-2]);
            //     W[0] = W_temp[0] - n/update_value.size * sum(d_w * update_value);
            //     d_w = -W_temp[0] * stack_f[stack_f.size()-1]/(stack_f[stack_f.size()-2] * pow(W[1], 2));
            //     W[1] = W_temp[1] - n/update_value.size * sum(d_w * update_value); 
            // }
    };
}	

#endif
