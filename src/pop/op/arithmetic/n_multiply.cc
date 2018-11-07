/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_multiply.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeMultiply::NodeMultiply(vector<double> W0)
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

            /// Evaluates the node and updates the state states. 
            void NodeMultiply::evaluate(const Data& data, State& state)
            {
                ArrayXd x1 = state.pop<double>();
                ArrayXd x2 = state.pop<double>();
               
                state.push<double>(limited(W[0]*x1 * W[1]*x2));
            }

            /// Evaluates the node symbolically
            void NodeMultiply::eval_eqn(State& state)
            {
	            state.push<double>("(" + state.popStr<double>() + "*" + state.popStr<double>() + ")");
            }

            ArrayXd NodeMultiply::getDerivative(Trace& state, int loc)
            {
                ArrayXd& x1 = state.get<double>()[state.size<double>()-1];
                ArrayXd& x2 = state.get<double>()[state.size<double>()-2];
                
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
