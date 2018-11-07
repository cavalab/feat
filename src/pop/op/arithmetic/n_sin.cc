/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sin.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeSin::NodeSin(vector<double> W0)
            {
	            name = "sin";
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

            /// Evaluates the node and updates the state states. 
            void NodeSin::evaluate(const Data& data, State& state)
            {

                state.push<double>(limited(sin(W[0]*state.pop<double>())));
            }

            /// Evaluates the node symbolically
            void NodeSin::eval_eqn(State& state)
            {
                state.push<double>("sin(" + state.popStr<double>() + ")");
            }

            ArrayXd NodeSin::getDerivative(Trace& state, int loc)
            {
                ArrayXd& x = state.get<double>()[state.size<double>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return x * cos(W[0] * x);
                    case 0: // d/dx0
                    default:
                        return W[0] * cos(W[0] * x);
                } 
            }

            NodeSin* NodeSin::clone_impl() const { return new NodeSin(*this); }

            NodeSin* NodeSin::rnd_clone_impl() const { return new NodeSin(); }  
        }
    }
}
