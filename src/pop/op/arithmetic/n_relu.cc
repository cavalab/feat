/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_relu.h"

namespace FT{

    namespace Pop{
        namespace Op{ 	
            NodeRelu::NodeRelu(vector<double> W0)
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

            /// Evaluates the node and updates the state states. 
            void NodeRelu::evaluate(const Data& data, State& state)
            {
                ArrayXd x = state.pop<double>();
                ArrayXd res = (W[0] * x > 0).select(W[0]*x, ArrayXd::Zero(x.size())+0.01); 
                state.push<double>(res);
            }

            /// Evaluates the node symbolically
            void NodeRelu::eval_eqn(State& state)
            {
                state.push<double>("relu("+ state.popStr<double>() +")");         	
            }

            ArrayXd NodeRelu::getDerivative(Trace& state, int loc)
            {

                ArrayXd& x = state.get<double>()[state.size<double>()-1];
                
                switch (loc) {
                    case 1: // d/dW
                        return (x>0).select(x,ArrayXd::Zero(x.size())+0.01);
                    case 0: // d/dx
                    default:
                        return (x>0).select(W[0],ArrayXd::Zero(x.size())+0.01);
                } 
            }
            
            NodeRelu* NodeRelu::clone_impl() const { return new NodeRelu(*this); }

            NodeRelu* NodeRelu::rnd_clone_impl() const { return new NodeRelu(); }  
        }
    }
}
