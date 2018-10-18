/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MIN
#define NODE_MIN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeMin : public Node
            {
            	public:
            	
            		NodeMin();
            		    		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);
                    
                protected:
                    NodeMin* clone_impl() const override;

                    NodeMin* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
