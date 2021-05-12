/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_tanh.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeTanh::NodeTanh(vector<float> W0)
            {
                name = "tanh";
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
            void NodeTanh::evaluate(const Data& data, State& state)
            {
                state.push<float>(limited(tanh(W[0]*state.pop<float>())));
            }
            #else
            void NodeTanh::evaluate(const Data& data, State& state)
            {
                GPU_Tanh(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeTanh::eval_eqn(State& state)
            {
                state.push<float>("tanh(" + to_string(W[0], 4) + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeTanh::getDerivative(Trace& state, int loc)
            {
                ArrayXf numerator;
                ArrayXf denom;
                ArrayXf x = state.get<float>()[state.size<float>()-1];
                switch (loc) {
                    case 1: // d/dw0
                        numerator = 4 * x * exp(2 * this->W[0] * x);
                        denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                        // numerator = 4 * x * exp(2 * this->W[0] * x - 1]); 
                        // denom = pow(exp(2 * this->W[0] * x) + 1,2);
                        return numerator/denom;
                    case 0: // d/dx0
                    default:
                        numerator = 4 * this->W[0] * exp(2 * this->W[0] * x);
                        denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                        // numerator = 4 * W[0] * exp(2 * W[0] * state.f[state.f.size() - 1]);
                        // denom = pow(exp(2 * W[0] * state.f[state.f.size()-1]),2);
                        return numerator/denom;
                } 
            }

            NodeTanh* NodeTanh::clone_impl() const { return new NodeTanh(*this); }
             
            NodeTanh* NodeTanh::rnd_clone_impl() const { return new NodeTanh(); }
        }
    }
}
