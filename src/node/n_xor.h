/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_XOR
#define NODE_XOR

#include "node.h"

namespace FT{
	class NodeXor : public Node
    {
    	public:
    	
    		NodeXor();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeXor* clone_impl() const override;
            
            NodeXor* rnd_clone_impl() const override;
    };
}	

#endif
