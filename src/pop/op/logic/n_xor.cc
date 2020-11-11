/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_xor.h"

namespace FT
{
    
    namespace Pop{
        namespace Op{    	
            NodeXor::NodeXor()
            {
                name = "xor";
	            otype = 'b';
	            arity['b'] = 2;
	            complexity = 2;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeXor::evaluate(const Data& data, State& state)
            {
	            ArrayXb x1 = state.pop<bool>();
                ArrayXb x2 = state.pop<bool>();

                ArrayXb res = (x1 != x2).select(ArrayXb::Ones(x1.size()), ArrayXb::Zero(x1.size()));

                state.push<bool>(res);
                
            }
            #else
            void NodeXor::evaluate(const Data& data, State& state)
            {
                GPU_Xor(state.dev_b, state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeXor::eval_eqn(State& state)
            {
                state.push<bool>("XOR(" + state.popStr<bool>() + "," 
                        + state.popStr<bool>() + ")");
            }
            
            NodeXor* NodeXor::clone_impl() const { return new NodeXor(*this); }

            NodeXor* NodeXor::rnd_clone_impl() const { return new NodeXor(); }  
        }
    }

}
