/* FEWTWO
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
    	
    		NodeNot();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
        protected:
            NodeNot* clone_impl() const override;
            NodeNot* rnd_clone_impl() const override;
    };
}	

#endif
