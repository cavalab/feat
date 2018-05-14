/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_OR
#define NODE_OR

#include "node.h"

namespace FT{
	class NodeOr : public Node
    {
    	public:
    	
    		NodeOr()
            {
                name = "or";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                stack.b.push(stack.b.pop() || stack.b.pop());

            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.bs.pop() + " OR " + stack.bs.pop() + ")");
            }
        protected:
            NodeOr* clone_impl() const override { return new NodeOr(*this); };  
            NodeOr* rnd_clone_impl() const override { return new NodeOr(); };  
    };
    
}	

#endif
