/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_STEP
#define NODE_STEP

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
        	class NodeStep : public Node
            {
            	public:
            	
            		NodeStep();
            		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);
                    
                protected:
                    NodeStep* clone_impl() const override;  
                    NodeStep* rnd_clone_impl() const override;  
            };
        }
    }
}	

#endif
