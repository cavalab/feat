/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_lessthan.h"

namespace FT{

    namespace Pop{
        namespace Op{
        
            NodeLessThan::NodeLessThan()
            {
	            name = "<";
	            otype = 'b';
	            arity['f'] = 2;
	            complexity = 2;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeLessThan::evaluate(const Data& data, State& state)
            {
                ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
                state.push<bool>(x1 < x2);
            }
            #else
            void NodeLessThan::evaluate(const Data& data, State& state)
            {
                GPU_LessThan(state.dev_f, state.dev_b, state.idx['f'], state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeLessThan::eval_eqn(State& state)
            {
                state.push<bool>("(" + state.popStr<float>() + "<" + state.popStr<float>() + ")");
            }
            
            NodeLessThan* NodeLessThan::clone_impl() const { return new NodeLessThan(*this); }

            NodeLessThan* NodeLessThan::rnd_clone_impl() const { return new NodeLessThan(); }  
            
        }
    }
}
