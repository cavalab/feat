/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "../node.h"

namespace FT{
	class NodeSign : public Node
    {
    	public:
    	
    		NodeSign();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

        protected:
            NodeSign* clone_impl() const override;  
            NodeSign* rnd_clone_impl() const override;  
    };
}	

#endif
