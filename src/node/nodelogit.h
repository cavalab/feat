/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOGIT
#define NODE_LOGIT

#include "nodeDx.h"

namespace FT{
	class NodeLogit : public NodeDx
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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(1/(1+(limited(exp(-W[0]*stack.f.pop())))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("1/(1+exp(-1*" + stack.fs.pop() + "))");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                ArrayXd numerator, denom;
                switch (loc) {
                    case 1: // d/dw0
                        numerator = stack_f[stack_f.size() -1] * exp(-W[0] * stack_f[stack_f.size() -1]);
                        denom = pow(1 + exp(-W[0] * stack_f[stack_f.size()-1]), 2);
                        return numerator/denom;
                    case 0: // d/dx0
                    default:
                        numerator = W[0] * exp(-W[0] * stack_f[stack_f.size() - 1]);
                        denom = pow(1 + exp(-W[0] * stack_f[stack_f.size() - 1]), 2);
                        return numerator/denom;
                } 
            }

        protected:
            NodeLogit* clone_impl() const override { return new NodeLogit(*this); };  
    };
}	

#endif
