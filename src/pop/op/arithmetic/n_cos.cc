/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_cos.h"

namespace FT{
	

    namespace Pop{
        namespace Op{
            NodeCos::NodeCos(vector<float> W0)
            {
	            name = "cos";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 3;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
            }

            #ifndef USE_CUDA    
            /// Evaluates the node and updates the state states. 
            void NodeCos::evaluate(const Data& data, State& state)
            {
                state.push<float>(limited(cos(W[0] * state.pop<float>())));
            }
            #else
            void NodeCos::evaluate(const Data& data, State& state)
            {
                GPU_Cos(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeCos::eval_eqn(State& state)
            {
                state.push<float>("cos(" + to_string(W[0], 4) + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeCos::getDerivative(Trace& state, int loc) {
            
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return x * -sin(W[0] * x);
                    case 0: // d/dx0
                    default:
                       return W[0] * -sin(W[0] * x);
                } 
            }
            
            NodeCos* NodeCos::clone_impl() const { return new NodeCos(*this); }
              
            NodeCos* NodeCos::rnd_clone_impl() const { return new NodeCos(); }
        }
    }
}
