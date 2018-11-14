/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_log.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeLog::NodeLog(vector<double> W0)
            {
	            name = "log";
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
            /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
            void NodeLog::evaluate(const Data& data, State& state)
            {
	            ArrayXd x = state.pop<double>();
                state.push<double>( (abs(x) > NEAR_ZERO).select(log(abs(W[0] * x)),MIN_DBL));
            }
            #else
            void NodeLog::evaluate(const Data& data, State& state)
            {
                GPU_Log(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeLog::eval_eqn(State& state)
            {
                state.push<double>("log(" + state.popStr<double>() + ")");
            }

            ArrayXd NodeLog::getDerivative(Trace& state, int loc)
            {
                ArrayXd& x = state.get<double>()[state.size<double>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return limited(1/(W[0] * ArrayXd::Ones(x.size())));
                    case 0: // d/dx0
                    default:
                        return limited(1/x);
                } 
            }
            
            NodeLog* NodeLog::clone_impl() const { return new NodeLog(*this); }

            NodeLog* NodeLog::rnd_clone_impl() const { return new NodeLog(); }  
        }
    }
}
