/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GEQ
#define NODE_GEQ

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeGEQ : public Node
            {
            	public:
            	
           		    NodeGEQ();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeGEQ* clone_impl() const override;
              
                    NodeGEQ* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
