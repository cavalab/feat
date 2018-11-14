/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEDIAN
#define NODE_MEDIAN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeMedian : public Node
            {
            	public:
            	
            		NodeMedian();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeMedian* clone_impl() const override;

                    NodeMedian* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
