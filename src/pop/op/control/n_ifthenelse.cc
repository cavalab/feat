/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_ifthenelse.h"

namespace FT{


    namespace Pop{
        namespace Op{
            NodeIfThenElse::NodeIfThenElse()
            {
		        name = "ite";
		        otype = 'f';
		        arity['f'] = 2;
		        arity['b'] = 1;
		        complexity = 5;
                W = {0.0, 0.0};
	        }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeIfThenElse::evaluate(const Data& data, State& state)
            {
                ArrayXf f1 = state.pop<float>();
                ArrayXf f2 = state.pop<float>();
                state.push<float>(limited(state.pop<bool>().select(f1,f2)));
            }
            #else
            void NodeIfThenElse::evaluate(const Data& data, State& state)
            {
                GPU_IfThenElse(state.dev_f, state.dev_b, state.idx[otype], state.idx['b'], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeIfThenElse::eval_eqn(State& state)
            {
                state.push<float>("if-then-else(" + state.popStr<bool>() + 
                                   "," + state.popStr<float>() + "," + 
                                   state.popStr<float>() + ")");
            }
            
            ArrayXf NodeIfThenElse::getDerivative(Trace& state, int loc) 
            {
                ArrayXf& xf = state.get<float>()[state.size<float>()-1];
                ArrayXb& xb = state.get<bool>()[state.size<bool>()-1];
                
                switch (loc) {
                    case 3: // d/dW[0]
                    case 2: 
                        return ArrayXf::Zero(xf.size()); 
                    case 1: // d/dx2
                        return (!xb).cast<float>(); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<float>(); 
                        /* .select(ArrayXf::Ones(state.f[state.f.size()-1].size(), */
                        /*                  ArrayXf::Zero(state.f[state.f.size()-1].size()); */
                } 
            }

            
            NodeIfThenElse* NodeIfThenElse::clone_impl() const { return new NodeIfThenElse(*this); }

            NodeIfThenElse* NodeIfThenElse::rnd_clone_impl() const { return new NodeIfThenElse(); }
        }
    }
}
