/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_VARIANCE
#define NODE_VARIANCE

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeVar : public Node
            {
            	public:
            	
            		NodeVar();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeVar* clone_impl() const override; 
                    NodeVar* rnd_clone_impl() const override; 
            };
        }
    }
}	

#endif
