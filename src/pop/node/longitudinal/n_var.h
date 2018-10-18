/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_VARIANCE
#define NODE_VARIANCE

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
        	class NodeVar : public Node
            {
            	public:
            	
            		NodeVar();
            		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);
                    
                protected:
                    NodeVar* clone_impl() const override; 
                    NodeVar* rnd_clone_impl() const override; 
            };
        }
    }
}	

#endif
