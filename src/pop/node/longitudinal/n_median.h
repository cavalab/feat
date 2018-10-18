/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEDIAN
#define NODE_MEDIAN

#include "../node.h"

namespace FT{
	class NodeMedian : public Node
    {
    	public:
    	
    		NodeMedian();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeMedian* clone_impl() const override;

            NodeMedian* rnd_clone_impl() const override;
    };
}	

#endif
