/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEAN
#define NODE_MEAN

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeMean : public Node
            {
            	public:
            	
            		NodeMean();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);
                    
                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeMean* clone_impl() const override;

                    NodeMean* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
