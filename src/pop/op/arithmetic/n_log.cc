/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_log.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeLog::NodeLog(vector<float> W0)
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
            /// Safe log: pushes log(abs(x)) or MIN_FLT if x is near zero. 
            void NodeLog::evaluate(const Data& data, State& state)
            {
	            ArrayXf x = state.pop<float>();
                state.push<float>( (abs(x) > NEAR_ZERO).select(log(abs(W[0] * x)),MIN_FLT));
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
                state.push<float>("log(" + to_string(W[0], 4) + "*" + state.popStr<float>() + ")");
            }

            ArrayXf NodeLog::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x = state.get<float>()[state.size<float>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return limited(1/(W[0] * ArrayXf::Ones(x.size())));
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
