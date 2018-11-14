/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LESSTHAN
#define NODE_LESSTHAN

#include "../node.h"

namespace FT{


    namespace Pop{
        namespace Op{
	        class NodeLessThan : public Node
            {
            	public:
            	
            		NodeLessThan();

                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeLessThan* clone_impl() const override;

                    NodeLessThan* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
