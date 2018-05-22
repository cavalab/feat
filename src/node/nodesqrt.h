/* FEAT
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
    	
    		NodeSqrt(vector<double> W0 = vector<double>())
            {
                name = "sqrt";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 2;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                stack.f.push(sqrt(W[0]*stack.f.pop().abs()));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("sqrt(|" + stack.fs.pop() + "|)");
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
            NodeSqrt* rnd_clone_impl() const override { return new NodeSqrt(); };  
    };
}	

#endif
