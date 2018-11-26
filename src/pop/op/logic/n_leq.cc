/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_leq.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeLEQ::NodeLEQ()
            {
	            name = "<=";
	            otype = 'b';
	            arity['f'] = 2;
	            complexity = 2;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeLEQ::evaluate(const Data& data, State& state)
            {
              	ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
                state.push<bool>(x1 <= x2);
            }
            #else
            void NodeLEQ::evaluate(const Data& data, State& state)
            {
                GPU_LEQ(state.dev_f, state.dev_b, state.idx['f'], state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeLEQ::eval_eqn(State& state)
            {
                state.push<bool>("(" + state.popStr<float>() + "<=" + state.popStr<float>() + ")");
            }
            
            NodeLEQ* NodeLEQ::clone_impl() const { return new NodeLEQ(*this); }  

            NodeLEQ* NodeLEQ::rnd_clone_impl() const { return new NodeLEQ(); }
        }
    }
}
