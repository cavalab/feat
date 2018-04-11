/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EXPONENTIAL
#define NODE_EXPONENTIAL

#include "nodeDx.h"

namespace FT{
	class NodeExponential : public NodeDx
    {
    	public:
   	
    		NodeExponential()
    		{
    			name = "exp";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
           		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(exp(W[0] * x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("exp(" + x + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] * limited(exp(W[0] * stack_f[stack_f.size()-1]));
                    case 0: // d/dx0
                    default:
                       return W[0] * limited(exp(W[0] * stack_f[stack_f.size()-1]));
                } 
            }
        protected:
            NodeExponential* clone_impl() const override { return new NodeExponential(*this); };  
    };
}	

#endif
