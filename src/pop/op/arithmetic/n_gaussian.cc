/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_gaussian.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeGaussian::NodeGaussian(vector<float> W0)
            {
                name = "gauss";
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

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeGaussian::evaluate(const Data& data, State& state)
            {
                state.push<float>(limited(exp(-pow(W[0] - state.pop<float>(), 2))));
            }
            #else
            void NodeGaussian::evaluate(const Data& data, State& state)
            {
                GPU_Gaussian(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeGaussian::eval_eqn(State& state)
            {
                state.push<float>("gauss(" + state.popStr<float>() + ")");
            }

            ArrayXf NodeGaussian::getDerivative(Trace& state, int loc) 
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
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
