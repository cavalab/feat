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
                ArrayXd f1 = state.pop<double>();
                ArrayXd f2 = state.pop<double>();
                state.push<double>(limited(state.pop<bool>().select(f1,f2)));
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
                state.push<double>("if-then-else(" + state.popStr<bool>() + 
                                   "," + state.popStr<double>() + "," + 
                                   state.popStr<double>() + ")");
            }
            
            ArrayXd NodeIfThenElse::getDerivative(Trace& state, int loc) 
            {
                ArrayXd& xf = state.get<double>()[state.size<double>()-1];
                ArrayXb& xb = state.get<bool>()[state.size<bool>()-1];
                
                switch (loc) {
                    case 3: // d/dW[0]
                    case 2: 
                        return ArrayXd::Zero(xf.size()); 
                    case 1: // d/dx2
                        return (!xb).cast<double>(); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<double>(); 
                        /* .select(ArrayXd::Ones(state.f[state.f.size()-1].size(), */
                        /*                  ArrayXd::Zero(state.f[state.f.size()-1].size()); */
                } 
            }

            
            NodeIfThenElse* NodeIfThenElse::clone_impl() const { return new NodeIfThenElse(*this); }

            NodeIfThenElse* NodeIfThenElse::rnd_clone_impl() const { return new NodeIfThenElse(); }
        }
    }
}
