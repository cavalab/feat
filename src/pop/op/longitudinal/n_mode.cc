/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_mode.h"

namespace FT{

    namespace Pop{
        namespace Op{    	
            NodeMode::NodeMode()
            {
                name = "mode";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeMode::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).mean();
                    
                state.z.pop();

                state.push<float>(tmp);
                
            }
            #else
            void NodeMode::evaluate(const Data& data, State& state)
            {
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    state.f.row(state.idx['f']) = state.z.top().first[x].mean();
                    
                state.z.pop();

                
           }
           #endif

            /// Evaluates the node symbolically
            void NodeMode::eval_eqn(State& state)
            {
                state.push<float>("mean(" + state.zs.pop() + ")");
            }
            
            NodeMode* NodeMode::clone_impl() const { return new NodeMode(*this); }

            NodeMode* NodeMode::rnd_clone_impl() const { return new NodeMode(); }  
        }
    }
}
