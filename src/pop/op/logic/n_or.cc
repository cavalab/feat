/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_or.h"
    	
namespace FT{

    
    namespace Pop{
        namespace Op{
            NodeOr::NodeOr()
            {
                name = "or";
	            otype = 'b';
	            arity['b'] = 2;
	            complexity = 2;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeOr::evaluate(const Data& data, State& state)
            {
                state.push<bool>(state.pop<bool>() || state.pop<bool>());

            }
            #else
            void NodeOr::evaluate(const Data& data, State& state)
            {
                GPU_Or(state.dev_b, state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeOr::eval_eqn(State& state)
            {
                state.push<bool>("OR(" + state.popStr<bool>() + "," + state.popStr<bool>() + ")");
            }
            
            NodeOr* NodeOr::clone_impl() const { return new NodeOr(*this); }

            NodeOr* NodeOr::rnd_clone_impl() const { return new NodeOr(); }  
        }
    }

}
