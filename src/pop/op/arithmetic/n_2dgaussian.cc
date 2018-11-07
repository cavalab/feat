/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/


#include "n_2dgaussian.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace Op{
            Node2dGaussian::Node2dGaussian(vector<double> W0)
            {
                name = "gaussian2d";
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
                ArrayXd x1 = state.pop<double>();
                ArrayXd x2 = state.pop<double>();

                state.push<double>(limited(exp(-1*(pow(W[0]*(x1-x1.mean()), 2)/(2*variance(x1)) 
                              + pow(W[1]*(x2 - x2.mean()), 2)/variance(x2)))));
            }
            #endif

            /// Evaluates the node symbolically
            void Node2dGaussian::eval_eqn(State& state)
            {
                state.push<double>("gauss2d(" +state.popStr<double>()+ "," +state.popStr<double>()+ ")");
            }

            ArrayXd Node2dGaussian::getDerivative(Trace& state, int loc) 
            {
                ArrayXd& x = state.get<double>()[state.size<double>()-1];

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


