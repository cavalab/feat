/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GREATERTHAN
#define NODE_GREATERTHAN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
        	class NodeGreaterThan : public Node
            {
            	public:
            	   	
            		NodeGreaterThan();
            		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);

                protected:
                    NodeGreaterThan* clone_impl() const override;
              
                    NodeGreaterThan* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
