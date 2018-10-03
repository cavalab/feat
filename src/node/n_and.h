/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_AND
#define NODE_AND

#include "node.h"

namespace FT{
	class NodeAnd : public Node
    {
    	public:
    	
    		NodeAnd();
    		    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);
            
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeAnd* clone_impl() const override;
      
            NodeAnd* rnd_clone_impl() const override;
    };
}	

#endif
