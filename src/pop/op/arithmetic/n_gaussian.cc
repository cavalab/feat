/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_gaussian.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeGaussian::NodeGaussian(vector<double> W0)
            {
                name = "gaussian";
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
            void NodeGaussian::evaluate(const Data& data, State& state)
            {
                state.push<double>(limited(exp(-pow(W[0] - state.pop<double>(), 2))));
            }

            /// Evaluates the node symbolically
            void NodeGaussian::eval_eqn(State& state)
            {
                state.push<double>("gauss(" + state.popStr<double>() + ")");
            }

            ArrayXd NodeGaussian::getDerivative(Trace& state, int loc) 
            {
                ArrayXd& x = state.get<double>()[state.size<double>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return limited(2 * (x - W[0]) * exp(-pow(W[0] - x, 2)));
                    case 0: // d/dx0
                    default:
                        return limited(2 * (W[0] - x) * exp(-pow(W[0] - x, 2)));
                } 
            }
            
            NodeGaussian* NodeGaussian::clone_impl() const { return new NodeGaussian(*this); }
              
            NodeGaussian* NodeGaussian::rnd_clone_impl() const { return new NodeGaussian(); }  
        }
    }
}
