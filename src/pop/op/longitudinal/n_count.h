/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COUNT
#define NODE_COUNT

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeCount : public Node
            {
            	public:
            	
            		NodeCount();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeCount* clone_impl() const override;
             
                    NodeCount* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
