/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_exponent.h"
    	  	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeExponent::NodeExponent(vector<float> W0)
            {
	            name = "^";
	            otype = 'f';
	            arity['f'] = 2;
	            arity['b'] = 0;
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
            void NodeExponent::evaluate(const Data& data, State& state)
            {
	            ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();

                state.push<float>(limited(pow(this->W[0] * x1, 
                                               this->W[1] * x2)));
            }
            #else
            void NodeExponent::evaluate(const Data& data, State& state)
            {
                GPU_Exponent(state.dev_f, state.idx[otype], state.N, W[0], W[1]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeExponent::eval_eqn(State& state)
            {
                state.push<float>("(" + to_string(W[0], 4) + "*" + state.popStr<float>() + ")^(" 
                        + to_string(W[1], 4) + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeExponent::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x1 = state.get<float>()[state.size<float>()-1];
                ArrayXf& x2 = state.get<float>()[state.size<float>()-2];
                
                switch (loc) {
                    case 3: // Weight for the power
                        return limited(pow(this->W[0] * x1,
                                           this->W[1] * x2) * limited(log(this->W[0] * x1)) * x2);
                    case 2: // Weight for the base
                        return limited(this->W[1] * x2 * pow(this->W[0] * x1,
                                       this->W[1] * x2) / this->W[0]);
                    case 1: // Power
                        return limited(this->W[1]*pow(this->W[0] * x1,
                                       this->W[1] * x2) * limited(log(this->W[0] * x1)));
                    case 0: // Base
                    default:
                        return limited(this->W[1] * x2 * pow(this->W[0] * x1, this->W[1] * x2) / x1);
                } 
            }
            
            NodeExponent* NodeExponent::clone_impl() const { return new NodeExponent(*this); }
              
            NodeExponent* NodeExponent::rnd_clone_impl() const { return new NodeExponent(); } 
        }
    }
}
