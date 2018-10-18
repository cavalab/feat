/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_OPENBRACE
#define NODE_OPENBRACE

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
        	class NodeLEQ : public Node
            {
            	public:
            	
            		NodeLEQ();
            		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);
                    
                protected:
                    NodeLEQ* clone_impl() const override;

                    NodeLEQ* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
