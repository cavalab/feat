/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_kurtosis.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace Op{

            NodeKurtosis::NodeKurtosis()
            {
                name = "kurtosis";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeKurtosis::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = kurtosis(limited(state.z.top().first[x]));
                    
                state.z.pop();
                state.push<float>(tmp);
                
            }
            #else
            void NodeKurtosis::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = kurtosis(limited(state.z.top().first[x]));
                    
                state.z.pop();
                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);
                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeKurtosis::eval_eqn(State& state)
            {
                state.push<float>("kurtosis(" + state.zs.pop() + ")");
            }
            
            NodeKurtosis* NodeKurtosis::clone_impl() const { return new NodeKurtosis(*this); }

            NodeKurtosis* NodeKurtosis::rnd_clone_impl() const { return new NodeKurtosis(); } 
        }
    }
}
