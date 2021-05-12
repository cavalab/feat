/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_square.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeSquare::NodeSquare(vector<float> W0)
            {
	            name = "^2";
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
            void NodeSquare::evaluate(const Data& data, State& state)
            {
                state.push<float>(limited(pow(W[0]*state.pop<float>(),2)));
            }
            #else
            void NodeSquare::evaluate(const Data& data, State& state)
            {
                GPU_Square(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSquare::eval_eqn(State& state)
            {
                state.push<float>("((" + to_string(W[0], 4) + "*"
                                  + state.popStr<float>() + ")^2)");
            }

            ArrayXf NodeSquare::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                switch (loc) {
                    case 1: // d/dw0
                        return 2 * pow(x, 2) * this->W[0];
                    case 0: // d/dx0
                    default:
                       return 2 * pow(this->W[0], 2) * x;
                } 
            }

            NodeSquare* NodeSquare::clone_impl() const { return new NodeSquare(*this); }

            NodeSquare* NodeSquare::rnd_clone_impl() const { return new NodeSquare(); }  
        }
    }
}
