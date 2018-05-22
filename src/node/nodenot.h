/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_NOT
#define NODE_NOT

#include "node.h"

namespace FT{
	class NodeNot : public Node
    {
    	public:
    	
    		NodeNot()
       		{
    			name = "not";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 1;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                stack.b.push(!stack.b.pop());
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("NOT(" + stack.bs.pop() + ")");
            }
        protected:
            NodeNot* clone_impl() const override { return new NodeNot(*this); };  
            NodeNot* rnd_clone_impl() const override { return new NodeNot(); };  
    };
    
}	

#endif
