/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_relu.h"

namespace FT{

    namespace Pop{
        namespace Op{ 	
            NodeRelu::NodeRelu(vector<float> W0)
            {
	            name = "relu";
	            otype = 'f';
	            arity['f'] = 1;
	            arity['b'] = 0;
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
            void NodeRelu::evaluate(const Data& data, State& state)
            {
                ArrayXf x = state.pop<float>();
                ArrayXf res = (W[0] * x > 0).select(W[0]*x, ArrayXf::Zero(x.size())+0.01); 
                state.push<float>(res);
            }
            #else
            /// Evaluates the node and updates the state states. 
            void NodeRelu::evaluate(const Data& data, State& state)
            {
                GPU_Relu(state.dev_f, state.idx[otype], state.N, (float)W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeRelu::eval_eqn(State& state)
            {
                state.push<float>("relu(" + to_string(W[0], 4) + "*" + state.popStr<float>() +")");         	
            }

            ArrayXf NodeRelu::getDerivative(Trace& state, int loc)
            {

                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dW
                        return (x>0).select(x,ArrayXf::Zero(x.size())+0.01);
                    case 0: // d/dx
                    default:
                        return (x>0).select(W[0],ArrayXf::Zero(x.size())+0.01);
                } 
            }
            
            NodeRelu* NodeRelu::clone_impl() const { return new NodeRelu(*this); }

            NodeRelu* NodeRelu::rnd_clone_impl() const { return new NodeRelu(); }  
        }
    }
}
