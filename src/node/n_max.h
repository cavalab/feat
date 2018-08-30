/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MAX
#define NODE_MAX

#include "node.h"

namespace FT{
	class NodeMax : public Node
    {
    	public:
    	
    		NodeMax();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeMax* clone_impl() const override;

            NodeMax* rnd_clone_impl() const override;
    };
}	

#endif
