/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IF
#define NODE_IF

#include "node.h"

namespace FT{
	class NodeIf : public NodeDx
    {
    	public:
    	   	
    		NodeIf(vector<double> W0 = vector<double>())
    		{
    			name = "if";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 1;
    			complexity = 5;
                W.push_back(0);
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                stack.f.push(limited(stack.b.pop().select(stack.f.pop(),0)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
              stack.fs.push("if(" + stack.bs.pop() + "," + stack.fs.pop() + "," + "0)");
            }
            
            ArrayXd getDerivative(Trace& stack, int loc) 
            {
                ArrayXb xb = stack.b[stack.b.size()-1];
                switch (loc) {
                    case 1: // d/dW[0]
                        return ArrayXd::Zero(stack.f[stack.f.size()-1].size()); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<double>(); 
                        /* .select(ArrayXd::Ones(stack.f[stack.f.size()-1].size(), */
                        /*                  ArrayXd::Zero(stack.f[stack.f.size()-1].size()); */
                } 
            }

        protected:
            NodeIf* clone_impl() const override { return new NodeIf(*this); };  
            NodeIf* rnd_clone_impl() const override { return new NodeIf(); };  
    };
}	

#endif
