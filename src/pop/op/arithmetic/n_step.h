/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_STEP
#define NODE_STEP

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeStep : public Node
            {
            	public:
            	
            		NodeStep();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeStep* clone_impl() const override;  
                    NodeStep* rnd_clone_impl() const override;  
            };
        }
    }
}	

#endif
