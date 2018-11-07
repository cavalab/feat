/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_OPENBRACE
#define NODE_OPENBRACE

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeLEQ : public Node
            {
            	public:
            	
            		NodeLEQ();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeLEQ* clone_impl() const override;

                    NodeLEQ* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
