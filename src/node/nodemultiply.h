/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MULTIPLY
#define NODE_MULTIPLY

#include "nodeDx.h"

namespace FT{
	class NodeMultiply : public NodeDx
    {
    	public:
    	
    		NodeMultiply()
       		{
    			name = "*";
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
        		ArrayXd x2 = stack_f.back(); stack_f.pop_back();
                ArrayXd x1 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(this->W[1] * x1 * this->W[0] * x2));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x2 = stack_f.back(); stack_f.pop_back();
            	string x1 = stack_f.back(); stack_f.pop_back();
            	stack_f.push_back("(" + x1 + "*" + x2 + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 3: 
                        return stack_f[stack_f.size()-1] * this->W[0] * stack_f[stack_f.size()-2];
                    case 2: 
                        return stack_f[stack_f.size()-1] * this->W[1] * stack_f[stack_f.size()-2];
                    case 1:
                        return this->W[0] * this->W[1] * stack_f[stack_f.size() - 2];
                    case 0:
                    default:
                       return this->W[1] * this->W[0] * stack_f[stack_f.size() - 1];
                } 
            }

            // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     switch (loc) {
            //         case 1:
            //             gradients.push_back(W[0] * W[1] * stack_f[stack_f.size() - 2]);
            //             break;
            //         case 0:
            //         default:
            //             gradients.push_back(W[0] * W[1] * stack_f[stack_f.size() - 1]);
            //     } 
            // }

            // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     update_value = 1
            //     for(auto g : gradients) {
            //         update_value *= g;
            //     }
                 
            //     W_temp = W[:]
            //     d_w = stack_f[stack_f.size()-1] * W_temp[1] * stack_f[stack_f.size()-2];
            //     W[0] = W_temp[0] - n/update_value.size * sum(d_w * update_value);
            //     d_w = d_w = stack_f[stack_f.size()-1] * W_temp[0] * stack_f.size()[stack_f.size()-2];
            //     W[1] = W_temp[1] - n/update_value.size * sum(d_w * update_value); 
            // }
        protected:
            NodeMultiply* clone_impl() const override { return new NodeMultiply(*this); };  
    };
}	

#endif
