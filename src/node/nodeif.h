/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IF
#define NODE_IF

#include "node.h"

namespace FT{
	class NodeIf : public Node
    {
    	public:
    	   	
    		NodeIf();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeIf* clone_impl() const override;
      
            NodeIf* rnd_clone_impl() const override;
    };
}	

#endif
