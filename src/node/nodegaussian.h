/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "nodeDx.h"

namespace FT{
	class NodeGaussian : public NodeDx
    {
    	public:
    	
    		NodeGaussian()
            {
                name = "gaussian";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(r());
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(limited(exp(-1*limited(pow(W[0] * stack.f.pop(), 2)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("exp(-(" + stack.fs.pop() + " ^ 2))");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return -2 * W[0] * pow(stack_f[stack_f.size() - 1], 2) * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                    case 0: // d/dx0
                    default:
                        return -2 * pow(W[0], 2) * stack_f[stack_f.size() - 1] * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                } 
            }
        protected:
            NodeGaussian* clone_impl() const override { return new NodeGaussian(*this); };  
    };
}	

#endif
