/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_KURTOSIS
#define NODE_KURTOSIS

#include "node.h"

namespace FT{
	class NodeKurtosis : public Node
    {
    	public:
    	
    		NodeKurtosis();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeKurtosis* clone_impl() const override;

            NodeKurtosis* rnd_clone_impl() const override;
    };
}	

#endif
