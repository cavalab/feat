/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_subtract.h"
    	    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeSubtract::NodeSubtract(vector<double> W0)
            {
	            name = "-";
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
            void NodeSubtract::evaluate(const Data& data, State& state)
            {
                ArrayXd x1 = state.pop<double>();
                ArrayXd x2 = state.pop<double>();
                state.push<double>(limited(this->W[0]*x1 - this->W[1]*x2));
            }
            #else
            void NodeSubtract::evaluate(const Data& data, State& state)
            {
                GPU_Subtract(state.dev_f, state.idx[otype], state.N, W[0], W[1]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSubtract::eval_eqn(State& state)
            {
                state.push<double>("(" + state.popStr<double>() + "-" + state.popStr<double>() + ")");
            }

            ArrayXd NodeSubtract::getDerivative(Trace& state, int loc)
            {
                ArrayXd x1 = state.get<double>()[state.size<double>()-1];
                ArrayXd x2 = state.get<double>()[state.size<double>()-2];
                
                switch (loc) {
                    case 3: // d/dW[1]
                        return -x2;
                    case 2: // d/dW[0]
                        return x1;
                    case 1: // d/dx2
                        return -this->W[1] * ArrayXd::Ones(x2.size());
                    case 0: //d/dx1
                    default:
                       return this->W[0] * ArrayXd::Ones(x1.size());
                } 
            }

            NodeSubtract* NodeSubtract::clone_impl() const { return new NodeSubtract(*this); }

            NodeSubtract* NodeSubtract::rnd_clone_impl() const { return new NodeSubtract(); }  
        }
    }
}
