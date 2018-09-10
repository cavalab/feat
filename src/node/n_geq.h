/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GEQ
#define NODE_GEQ

#include "node.h"

namespace FT{
	class NodeGEQ : public Node
    {
    	public:
    	
   		    NodeGEQ();  
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

        protected:
            NodeGEQ* clone_impl() const override;
            
            NodeGEQ* rnd_clone_impl() const override;
    };
}	

#endif
