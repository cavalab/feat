/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_XOR
#define NODE_XOR

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeXor : public Node
            {
            	public:
            	
            		NodeXor();
            		    		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeXor* clone_impl() const override;

                    NodeXor* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
