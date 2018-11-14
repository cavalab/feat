/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_add.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeAdd::NodeAdd(vector<double> W0)
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
                ArrayXd x1 = state.pop<double>();
                ArrayXd x2 = state.pop<double>();
                state.push<double>(limited(this->W[0]*x1+this->W[1]*x2));
                /* state.push<double>(limited(this->W[0]*state.pop<double>()+this->W[1]*state.pop<double>())); */
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
                state.push<double>("(" + state.popStr<double>() + "+" + state.popStr<double>() + ")");
            }

            // NEED TO MAKE SURE CASE 0 IS TOP OF STACK, CASE 2 IS w[0]
            ArrayXd NodeAdd::getDerivative(Trace& state, int loc) 
            {
                ArrayXd x1 = state.get<double>()[state.size<double>()-1];
                ArrayXd x2 = state.get<double>()[state.size<double>()-2];
                
                switch (loc) {
                    case 3: // d/dW[1] 
                        return x2;
                    case 2: // d/dW[0]
                        return x1;
                    case 1: // d/dx2
                        return this->W[1] * ArrayXd::Ones(x2.size());
                    case 0: // d/dx1
                    default:
                        return this->W[0] * ArrayXd::Ones(x1.size());
                } 
            }
            
            NodeAdd* NodeAdd::clone_impl() const { return new NodeAdd(*this); }
              
            NodeAdd* NodeAdd::rnd_clone_impl() const { return new NodeAdd(); }
        }
    }
}
