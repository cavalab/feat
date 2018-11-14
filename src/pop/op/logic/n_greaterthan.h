/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GREATERTHAN
#define NODE_GREATERTHAN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeGreaterThan : public Node
            {
            	public:
            	   	
            		NodeGreaterThan();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                protected:
                    NodeGreaterThan* clone_impl() const override;
              
                    NodeGreaterThan* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
