/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_if.h"
    	
namespace FT{


    namespace Pop{
        namespace Op{       	
            NodeIf::NodeIf(vector<float> W0)
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
                state.push<float>(limited(state.pop<bool>().select(state.pop<float>(),0)));
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
              state.push<float>("if(" + state.popStr<bool>() + "," + state.popStr<float>() + "," + "0)");
            }
            
            ArrayXf NodeIf::getDerivative(Trace& state, int loc) 
            {
                ArrayXf& xf = state.get<float>()[state.size<float>()-1];
                ArrayXb& xb = state.get<bool>()[state.size<bool>()-1];
                
                switch (loc) {
                    case 1: // d/dW[0]
                        return ArrayXf::Zero(xf.size()); 
                    case 0: // d/dx1
                    default:
                        return xb.cast<float>(); 
                        /* .select(ArrayXf::Ones(state.f[state.f.size()-1].size(), */
                        /*                  ArrayXf::Zero(state.f[state.f.size()-1].size()); */
                } 
            }
            
            NodeIf* NodeIf::clone_impl() const { return new NodeIf(*this); }
              
            NodeIf* NodeIf::rnd_clone_impl() const { return new NodeIf(); }
        }
    }
}
