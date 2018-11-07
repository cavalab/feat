/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_logit.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeLogit::NodeLogit(vector<double> W0)
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

            /// Evaluates the node and updates the state states. 
            void NodeLogit::evaluate(const Data& data, State& state)
            {
                state.push<double>(1/(1+(limited(exp(-W[0]*state.pop<double>())))));
            }

            /// Evaluates the node symbolically
            void NodeLogit::eval_eqn(State& state)
            {
                /* state.push<double>("1/(1+exp(-" + state.popStr<double>() + "))"); */
                state.push<double>("logit(" + state.popStr<double>() + ")");
            }

            ArrayXd NodeLogit::getDerivative(Trace& state, int loc) 
            {
                ArrayXd numerator, denom;
                
                ArrayXd& x = state.get<double>()[state.size<double>()-1];
                
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
