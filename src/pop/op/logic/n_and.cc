/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_and.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeAnd::NodeAnd()
            {
	            name = "and";
	            otype = 'b';
	            arity['b'] = 2;
	            complexity = 2;
            }

            #ifndef USE_CUDA	
            /// Evaluates the node and updates the state states. 
            void NodeAnd::evaluate(const Data& data, State& state)
            {
                state.push<bool>(state.pop<bool>() && state.pop<bool>());

            }
            #else
            void NodeAnd::evaluate(const Data& data, State& state)
            {
               GPU_And(state.dev_b, state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeAnd::eval_eqn(State& state)
            {
                state.push<bool>("AND(" + state.popStr<bool>() + "," 
                                 + state.popStr<bool>() + ")");
            }
            
            NodeAnd* NodeAnd::clone_impl() const { return new NodeAnd(*this); }
              
            NodeAnd* NodeAnd::rnd_clone_impl() const { return new NodeAnd(); } 
        }
    }
}
