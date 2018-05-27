/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SLOPE
#define NODE_SLOPE

#include "node.h"

namespace FT{
	class NodeSlope : public Node
    {
    	public:
    	
    		NodeSlope();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeSlope* clone_impl() const override; 
            NodeSlope* rnd_clone_impl() const override; 
    };
}	

#endif
