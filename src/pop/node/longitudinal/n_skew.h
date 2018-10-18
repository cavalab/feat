/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SKEW
#define NODE_SKEW

#include "../node.h"

namespace FT{
	class NodeSkew : public Node
    {
    	public:
    	
    		NodeSkew();
    		    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeSkew* clone_impl() const override; 
            NodeSkew* rnd_clone_impl() const override; 
    };
}	

#endif
