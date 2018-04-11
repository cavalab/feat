/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COS
#define NODE_COS

#include "nodeDx.h"

namespace FT{
	class NodeCos : public NodeDx
    {
    	public:
    	  	
    		NodeCos()
    		{
    			name = "cos";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(cos(W[0] * x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("cos(" + x + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] * -sin(W[0] * stack_f[stack_f.size() - 1]);
                    case 0: // d/dx0
                    default:
                       return W[0] * -sin(W[0] * stack_f[stack_f.size() - 1]);
                } 
            }

            // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     switch (loc) {
            //         case 0:
            //         default:
            //             gradients.push_back(W[0] * -sin(W[0] * stack_f[stack_f.size() - 1]));
            //     } 
            // }

            // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
            //     update_value = 1
            //     for(auto g : gradients) {
            //         update_value *= g;
            //     }
                 
            //     d_w = stack_f[stack_f.size()-1] * -sin(W[0] * stack_f[stack_f-1])
            //     W[0] = W[0] - n/update_value.size * sum(d_w * update_value);
            // }

        protected:
            NodeCos* clone_impl() const override { return new NodeCos(*this); };  
    };
}	

#endif
