/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/


#include "n_2dgaussian.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace Op{
            Node2dGaussian::Node2dGaussian(vector<float> W0)
            {
                name = "gauss2d";
	            otype = 'f';
	            arity['f'] = 2;
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

            /// Evaluates the node and updates the state states. 
            #ifndef USE_CUDA
            void Node2dGaussian::evaluate(const Data& data, State& state)
            {
                ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
                
                state.push<float>(limited(exp(-1*(pow(W[0]*(x1-x1.mean()), 2)/(2*variance(x1)) 
                              + pow(W[1]*(x2 - x2.mean()), 2)/variance(x2)))));
            }
            #else
            /// Evaluates the node and updates the state states. 
            void Node2dGaussian::evaluate(const Data& data, State& state)
            {            
                
                ArrayXf x1(state.N);
                ArrayXf x2(state.N);
                
                state.copy_to_host(x1.data(), (state.idx['f']-1)*state.N);
                state.copy_to_host(x2.data(), (state.idx['f']-2)*state.N);

                float x1mean = x1.mean();
                float x2mean = x2.mean();
                
                float x1var = variance(x1);
                float x2var = variance(x2);
                              
                GPU_Gaussian2D(state.dev_f, state.idx[otype],
                               (float)x1mean, (float)x1var,
                               (float)x2mean, (float)x2var,
                               (float)W[0], (float)W[1], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void Node2dGaussian::eval_eqn(State& state)
            {
                state.push<float>("gauss2d(" +state.popStr<float>()+ "," +state.popStr<float>()+ ")");
            }

            ArrayXf Node2dGaussian::getDerivative(Trace& state, int loc) 
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];

                switch (loc) {
                    case 1: // d/dw0
                        return -2 * W[0] * pow(x, 2) * exp(-pow(W[0] * x, 2));
                    case 0: // d/dx0
                    default:
                        return -2 * pow(W[0], 2) * x * exp(-pow(W[0] * x, 2));
                } 
            }
            
            Node2dGaussian* Node2dGaussian::clone_impl() const { return new Node2dGaussian(*this); }  

            Node2dGaussian* Node2dGaussian::rnd_clone_impl() const { return new Node2dGaussian(); }
        }
    }
}


