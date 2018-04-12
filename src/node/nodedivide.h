/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#define NEAR_ZERO 0.00001  // Added in as placeholder, ask Bill what number this should be

#include "nodeDx.h"

namespace FT{
	class NodeDivide : public NodeDx
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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                ArrayXd x2 = stack.f.pop();
                ArrayXd x1 = stack.f.pop();
                // safe division returns x1/x2 if x2 != 0, and MAX_DBL otherwise               
                stack.f.push( (abs(x2) > NEAR_ZERO ).select((this->W[1] * x1) / (this->W[0] * x2), 
                                                            1.0) ); 
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "/" + stack.fs.pop() + ")");            	
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 3: // d/dw1
                        return stack_f[stack_f.size()-1]/(this->W[0] * stack_f[stack_f.size()-2]);
                    case 2: // d/dw0
                        return -this->W[1] * stack_f[stack_f.size()-1]/(stack_f[stack_f.size()-2] * pow(W[0], 2));
                    case 1: // d/dx1
                        return this->W[1]/(this->W[0] * stack_f[stack_f.size()-2]);
                    case 0: // d/dx0
                    default:
                       return -this->W[1] * stack_f[stack_f.size() - 1]/(this->W[0] * pow(stack_f[stack_f.size()], 2));
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
        protected:
            NodeDivide* clone_impl() const override { return new NodeDivide(*this); };  
    };
}	

#endif
