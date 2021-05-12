/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_logit.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeLogit::NodeLogit(vector<float> W0)
            {
                name = "logit";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 4;

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
            void NodeLogit::evaluate(const Data& data, State& state)
            {
                state.push<float>(1/(1+(limited(exp(-W[0]*state.pop<float>())))));
            }
            #else
            void NodeLogit::evaluate(const Data& data, State& state)
            {
                GPU_Logit(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeLogit::eval_eqn(State& state)
            {
                /* state.push<float>("1/(1+exp(-" + state.popStr<float>() + "))"); */
                state.push<float>("logit(" + to_string(W[0], 4) + "*"+ state.popStr<float>() + ")");
            }

            ArrayXf NodeLogit::getDerivative(Trace& state, int loc) 
            {
                ArrayXf numerator, denom;
                
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        numerator = x * exp(-W[0] * x);
                        denom = pow(1 + exp(-W[0] * x), 2);
                        return numerator/denom;
                    case 0: // d/dx0
                    default:
                        numerator = W[0] * exp(-W[0] * x);
                        denom = pow(1 + exp(-W[0] * x), 2);
                        return numerator/denom;
                } 
            }
            
            NodeLogit* NodeLogit::clone_impl() const { return new NodeLogit(*this); }

            NodeLogit* NodeLogit::rnd_clone_impl() const { return new NodeLogit(); }
        }
    }
}
