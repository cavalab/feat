/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_add.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeAdd::NodeAdd(vector<float> W0)
            {
	            name = "+";
	            otype = 'f';
	            arity['f'] = 2;
	            complexity = 1;

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
            void NodeAdd::evaluate(const Data& data, State& state)
            {
                ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
                state.push<float>(limited(this->W[0]*x1+this->W[1]*x2));
                /* state.push<float>(limited(this->W[0]*state.pop<float>()+this->W[1]*state.pop<float>())); */
            }
            #else
            void NodeAdd::evaluate(const Data& data, State& state)
	        {
                GPU_Add(state.dev_f, state.idx[otype], state.N, (float)W[0], (float)W[1]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeAdd::eval_eqn(State& state)
            {
                state.push<float>("(" + to_string(W[0], 4) + "*" + state.popStr<float>() + "+" 
                                    + to_string(W[1], 4) + "*" + state.popStr<float>() + ")");
            }

            // NEED TO MAKE SURE CASE 0 IS TOP OF STACK, CASE 2 IS w[0]
            ArrayXf NodeAdd::getDerivative(Trace& state, int loc) 
            {
                ArrayXf x1 = state.get<float>()[state.size<float>()-1];
                ArrayXf x2 = state.get<float>()[state.size<float>()-2];
                
                switch (loc) {
                    case 3: // d/dW[1] 
                        return x2;
                    case 2: // d/dW[0]
                        return x1;
                    case 1: // d/dx2
                        return this->W[1] * ArrayXf::Ones(x2.size());
                    case 0: // d/dx1
                    default:
                        return this->W[0] * ArrayXf::Ones(x1.size());
                } 
            }
            
            NodeAdd* NodeAdd::clone_impl() const { return new NodeAdd(*this); }
              
            NodeAdd* NodeAdd::rnd_clone_impl() const { return new NodeAdd(); }
        }
    }
}
