/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQRT
#define NODE_SQRT

#include "nodeDx.h"

namespace FT{
	class NodeSqrt : public NodeDx
    {
    	public:
    	
    		NodeSqrt()
            {
                name = "sqrt";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 2;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(sqrt(abs(this->W[0] * x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("sqrt(|" + x + "|)");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] / (2 * sqrt(this->W[0] * stack_f[stack_f.size()-1]));
                    case 0: // d/dx0
                    default:
                        return this->W[0] / (2 * sqrt(this->W[0] * stack_f[stack_f.size()-1]));
                } 
            }

        protected:
            NodeSqrt* clone_impl() const override { return new NodeSqrt(*this); };  
    };
}	

#endif
