/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_multiply.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeMultiply::NodeMultiply(vector<float> W0)
            {
	            name = "*";
	            otype = 'f';
	            arity['f'] = 2;
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
            void NodeMultiply::evaluate(const Data& data, State& state)
            {
                ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
               
                state.push<float>(limited(W[0]*x1 * W[1]*x2));
            }
            #else
            void NodeMultiply::evaluate(const Data& data, State& state)
            {
                GPU_Multiply(state.dev_f, state.idx[otype], state.N, W[0], W[1]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeMultiply::eval_eqn(State& state)
            {
	            state.push<float>("(" + to_string(W[0]*W[1], 4) + "*" 
                        + state.popStr<float>() + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeMultiply::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x1 = state.get<float>()[state.size<float>()-1];
                ArrayXf& x2 = state.get<float>()[state.size<float>()-2];
                
                switch (loc) {
                    case 3: // d/dW[1]
                        return x1 * this->W[0] * x2;
                    case 2: // d/dW[0] 
                        return x1 * this->W[1] * x2;
                    case 1: // d/dx2
                        return this->W[0] * this->W[1] * x1;
                    case 0: // d/dx1
                    default:
                        return this->W[1] * this->W[0] * x2;
                } 
            }
            
            NodeMultiply* NodeMultiply::clone_impl() const { return new NodeMultiply(*this); }

            NodeMultiply* NodeMultiply::rnd_clone_impl() const { return new NodeMultiply(); }  
            
        }
    }
}
