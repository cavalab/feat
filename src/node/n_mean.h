/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEAN
#define NODE_MEAN

#include "node.h"

namespace FT{
	class NodeMean : public Node
    {
    	public:
    	
    		NodeMean();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);
            
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeMean* clone_impl() const override;

            NodeMean* rnd_clone_impl() const override;
    };
}	

#endif
