/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "nodeDx.h"

namespace FT{
	class NodeCube : public NodeDx
    {
    	public:
    		  
    		NodeCube(vector<double> W0 = vector<double>())
    		{
    			name = "cube";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 33;

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
                stack.f.push(limited(pow(this->W[0] * stack.f.pop(),3)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "^3)");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return 3 * pow(stack_f[stack_f.size()-1], 3) * pow(this->W[0], 2);
                    case 0: // d/dx0
                    default:
                       return 3 * pow(this->W[0], 3) * pow(stack_f[stack_f.size()-1], 2);
                } 
            }
        protected:
            NodeCube* clone_impl() const override { return new NodeCube(*this); };  
            NodeCube* rnd_clone_impl() const override { return new NodeCube(); };  
    };
}	

#endif

