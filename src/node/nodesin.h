/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIN
#define NODE_SIN

#include "nodeDx.h"

namespace FT{
	class NodeSin : public NodeDx
    {
    	public:
    	
    		NodeSin(vector<double> W0 = vector<double>())
       		{
    			name = "sin";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;

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

                stack.f.push(limited(sin(W[0]*stack.f.pop())));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("sin(" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(Trace& stack, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return stack.f[stack.f.size() - 1] * cos(W[0] * stack.f[stack.f.size()-1]);
                    case 0: // d/dx0
                    default:
                        return W[0] * cos(W[0] * stack.f[stack.f.size() - 1]);
                } 
            }

        protected:
            NodeSin* clone_impl() const override { return new NodeSin(*this); };  
            NodeSin* rnd_clone_impl() const override { return new NodeSin(); };  
    };
}	

#endif
