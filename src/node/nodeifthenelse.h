/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "node.h"

namespace FT{
	class NodeIfThenElse : public NodeDx
    {
    	public:
    	
    		NodeIfThenElse()
    	    {
    			name = "ite";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 1;
    			complexity = 5;
                W = {0.0, 0.0};
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                ArrayXd f1 = stack.f.pop();
                ArrayXd f2 = stack.f.pop();
                stack.f.push(limited(stack.b.pop().select(f1,f2)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("if-then-else(" + stack.bs.pop() + "," + stack.fs.pop() + "," + stack.fs.pop() + ")");
            }
            
            ArrayXd getDerivative(Trace& stack, int loc) 
            {
                ArrayXb xb = stack.b[stack.b.size()-1];
                switch (loc) {
                    case 3: // d/dW[0]
                    case 2: 
                        return ArrayXd::Zero(stack.f[stack.f.size()-1].size()); 
                    case 1: // d/dx2
                        return (!xb).cast<double>(); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<double>(); 
                        /* .select(ArrayXd::Ones(stack.f[stack.f.size()-1].size(), */
                        /*                  ArrayXd::Zero(stack.f[stack.f.size()-1].size()); */
                } 
            }
        protected:
            NodeIfThenElse* clone_impl() const override { return new NodeIfThenElse(*this); };  
            NodeIfThenElse* rnd_clone_impl() const override { return new NodeIfThenElse(); };  
    };

}	

#endif
