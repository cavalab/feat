/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_not.h"
    	
namespace FT{


    namespace Pop{
        namespace Op{
            NodeNot::NodeNot()
            {
	            name = "not";
	            otype = 'b';
	            arity['b'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeNot::evaluate(const Data& data, State& state)
            {
                state.push<bool>(!state.pop<bool>());
            }
            #else
            void NodeNot::evaluate(const Data& data, State& state)
            {
                GPU_Not(state.dev_b, state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeNot::eval_eqn(State& state)
            {
                state.push<bool>("NOT(" + state.popStr<bool>() + ")");
            }
            
            NodeNot* NodeNot::clone_impl() const { return new NodeNot(*this); }

            NodeNot* NodeNot::rnd_clone_impl() const { return new NodeNot(); }  
        }
    }
}
