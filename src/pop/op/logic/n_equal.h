/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EQUAL
#define NODE_EQUAL

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeEqual : public Node
            {
            	public:
            	   	
            		NodeEqual();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeEqual* clone_impl() const override;

                    NodeEqual* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
