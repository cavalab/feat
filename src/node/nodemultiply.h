/* FEAT
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
    	
    		NodeMultiply(vector<double> W0 = vector<double>())
       		{
    			name = "*";
    			otype = 'f';
    			arity['f'] = 2;
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
                stack.f.push(limited(W[0]*stack.f.pop() * W[1]*stack.f.pop()));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
            	stack.fs.push("(" + stack.fs.pop() + "*" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(Trace& stack, int loc) {
                switch (loc) {
                    case 3: // d/dW[1]
                        return stack.f[stack.f.size()-1] * this->W[0] * stack.f[stack.f.size()-2];
                    case 2: // d/dW[0] 
                        return stack.f[stack.f.size()-1] * this->W[1] * stack.f[stack.f.size()-2];
                    case 1: // d/dx2
                        return this->W[0] * this->W[1] * stack.f[stack.f.size() - 2];
                    case 0: // d/dx1
                    default:
                        return this->W[1] * this->W[0] * stack.f[stack.f.size() - 1];
                } 
            }

        protected:
            NodeMultiply* clone_impl() const override { return new NodeMultiply(*this); };  
            NodeMultiply* rnd_clone_impl() const override { return new NodeMultiply(); };  
    };
}	

#endif
