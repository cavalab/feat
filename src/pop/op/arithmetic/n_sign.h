/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeSign : public Node
            {
            	public:
            	
            		NodeSign();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                protected:
                    NodeSign* clone_impl() const override;  
                    NodeSign* rnd_clone_impl() const override;  
            };
        }
    }
}	

#endif
