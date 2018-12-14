/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_var.h"
#include "../../../util/utils.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeVar::NodeVar()
            {
                name = "variance";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeVar::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = variance(limited(state.z.top().first[x]));
                    
                state.z.pop();

                state.push<float>(tmp);
                
            }
            #else
            void NodeVar::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = variance(limited(state.z.top().first[x]));
                    
                state.z.pop();

                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);
                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeVar::eval_eqn(State& state)
            {
                state.push<float>("variance(" + state.zs.pop() + ")");
            }

            NodeVar* NodeVar::clone_impl() const { return new NodeVar(*this); }

            NodeVar* NodeVar::rnd_clone_impl() const { return new NodeVar(); } 
        }
    }

}
