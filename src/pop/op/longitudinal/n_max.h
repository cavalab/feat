/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MAX
#define NODE_MAX

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeMax : public Node
            {
            	public:
            	
            		NodeMax();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeMax* clone_impl() const override;

                    NodeMax* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
