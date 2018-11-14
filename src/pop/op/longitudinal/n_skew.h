/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SKEW
#define NODE_SKEW

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeSkew : public Node
            {
            	public:
            	
            		NodeSkew();
            		    		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeSkew* clone_impl() const override; 
                    NodeSkew* rnd_clone_impl() const override; 
            };
        }
    }
}	

#endif
