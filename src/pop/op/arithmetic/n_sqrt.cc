/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sqrt.h"

namespace FT{

    namespace Pop{
        namespace Op{    	
            NodeSqrt::NodeSqrt(vector<float> W0)
            {
                name = "sqrt";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 2;

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
            void NodeSqrt::evaluate(const Data& data, State& state)
            {
                state.push<float>(sqrt(W[0]*state.pop<float>().abs()));
            }
            #else
            void NodeSqrt::evaluate(const Data& data, State& state)
            {
                GPU_Sqrt(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSqrt::eval_eqn(State& state)
            {
                state.push<float>("sqrt(|" + to_string(W[0], 4) + "*" 
                                  + state.popStr<float>() + "|)");
            }

            ArrayXf NodeSqrt::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return limited(x / (2 * sqrt(this->W[0] * x)));
                    case 0: // d/dx0
                    default:
                        return limited(this->W[0] / (2 * sqrt(this->W[0] * x)));
                } 
            }

            NodeSqrt* NodeSqrt::clone_impl() const { return new NodeSqrt(*this); }

            NodeSqrt* NodeSqrt::rnd_clone_impl() const { return new NodeSqrt(); }  
        }
    }
}
