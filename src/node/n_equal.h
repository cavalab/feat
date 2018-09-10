/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EQUAL
#define NODE_EQUAL

#include "node.h"

namespace FT{
	class NodeEqual : public Node
    {
    	public:
    	   	
    		NodeEqual();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeEqual* clone_impl() const override;
            
            NodeEqual* rnd_clone_impl() const override;
    };
}	

#endif
