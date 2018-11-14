/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_if.h"
    	
namespace FT{


    namespace Pop{
        namespace Op{       	
            NodeIf::NodeIf(vector<double> W0)
            {
		        name = "if";
		        otype = 'f';
		        arity['f'] = 1;
		        arity['b'] = 1;
		        complexity = 5;
                W.push_back(0);
	        }


            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeIf::evaluate(const Data& data, State& state)
            {
                state.push<double>(limited(state.pop<bool>().select(state.pop<double>(),0)));
            }
            #else
            void NodeIf::evaluate(const Data& data, State& state)
            {
                GPU_If(state.dev_f, state.dev_b, state.idx['f'], state.idx['b'], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeIf::eval_eqn(State& state)
            {
              state.push<double>("if(" + state.popStr<bool>() + "," + state.popStr<double>() + "," + "0)");
            }
            
            ArrayXd NodeIf::getDerivative(Trace& state, int loc) 
            {
                ArrayXd& xf = state.get<double>()[state.size<double>()-1];
                ArrayXb& xb = state.get<bool>()[state.size<bool>()-1];
                
                switch (loc) {
                    case 1: // d/dW[0]
                        return ArrayXd::Zero(xf.size()); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<double>(); 
                        /* .select(ArrayXd::Ones(state.f[state.f.size()-1].size(), */
                        /*                  ArrayXd::Zero(state.f[state.f.size()-1].size()); */
                } 
            }
            
            NodeIf* NodeIf::clone_impl() const { return new NodeIf(*this); }
              
            NodeIf* NodeIf::rnd_clone_impl() const { return new NodeIf(); }
        }
    }
}
