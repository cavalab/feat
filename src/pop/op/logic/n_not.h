/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_NOT
#define NODE_NOT

#include "../node.h"

namespace FT{

    
    namespace Pop{
        namespace Op{
	        class NodeNot : public Node
            {
            	public:
            	
            		NodeNot();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeNot* clone_impl() const override;

                    NodeNot* rnd_clone_impl() const override;
            };
        }
    }
    
}	

#endif
