/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LESSTHAN
#define NODE_LESSTHAN

#include "node.h"

namespace FT{
	class NodeLessThan : public Node
    {
    	public:
    	
    		NodeLessThan()
       		{
    			name = "<";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}

            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                ArrayXd x1 = stack.f.pop();
                ArrayXd x2 = stack.f.pop();
                stack.b.push(x1 < x2);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.fs.pop() + "<" + stack.fs.pop() + ")");
            }
        protected:
            NodeLessThan* clone_impl() const override { return new NodeLessThan(*this); };  
            NodeLessThan* rnd_clone_impl() const override { return new NodeLessThan(); };  
    };
}	

#endif
