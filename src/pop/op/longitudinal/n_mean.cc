/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_mean.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeMean::NodeMean()
            {
                name = "mean";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA 
            /// Evaluates the node and updates the state states. 
            void NodeMean::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).mean();
                  
                state.z.pop();
                
                state.push<float>(tmp);
                
            }
            #else
            void NodeMean::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).mean();
                  
                state.z.pop();
                
                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeMean::eval_eqn(State& state)
            {
                state.push<float>("mean(" + state.zs.pop() + ")");
            }
            
            NodeMean* NodeMean::clone_impl() const { return new NodeMean(*this); }

            NodeMean* NodeMean::rnd_clone_impl() const { return new NodeMean(); } 
        }
    }
}
