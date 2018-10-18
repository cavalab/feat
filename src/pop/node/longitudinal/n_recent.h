/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_RECENT
#define NODE_RECENT 

#include "../node.h"

namespace FT{
	class NodeRecent : public Node
    {
    	public:
    	
    		NodeRecent();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeRecent* clone_impl() const override;

            NodeRecent* rnd_clone_impl() const override;
    };
}	

#endif
