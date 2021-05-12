/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sin.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeSin::NodeSin(vector<float> W0)
            {
	            name = "sin";
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
            void NodeSin::evaluate(const Data& data, State& state)
            {

                state.push<float>(limited(sin(W[0]*state.pop<float>())));
            }
            #else
            void NodeSin::evaluate(const Data& data, State& state)
            {
                GPU_Sin(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSin::eval_eqn(State& state)
            {
                state.push<float>("sin(" + to_string(W[0], 4) + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeSin::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return x * cos(W[0] * x);
                    case 0: // d/dx0
                    default:
                        return W[0] * cos(W[0] * x);
                } 
            }

            NodeSin* NodeSin::clone_impl() const { return new NodeSin(*this); }

            NodeSin* NodeSin::rnd_clone_impl() const { return new NodeSin(); }  
        }
    }
}
