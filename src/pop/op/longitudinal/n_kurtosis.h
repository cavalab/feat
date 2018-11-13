/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_KURTOSIS
#define NODE_KURTOSIS

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeKurtosis : public Node
            {
            	public:
            	
            		NodeKurtosis();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeKurtosis* clone_impl() const override;

                    NodeKurtosis* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
