/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SLOPE
#define NODE_SLOPE

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeSlope : public Node
            {
            	public:
            	
            		NodeSlope();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    float slope(const ArrayXf& x, const ArrayXf& y);

                protected:
                    NodeSlope* clone_impl() const override; 
                    NodeSlope* rnd_clone_impl() const override; 
            };
        }
    }
}	

#endif
