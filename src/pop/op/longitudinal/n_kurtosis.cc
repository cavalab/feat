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

            /// Evaluates the node and updates the state states. 
            void NodeKurtosis::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = kurtosis(limited(state.z.top().first[x]));
                    
                state.z.pop();
                state.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeKurtosis::eval_eqn(State& state)
            {
                state.push<double>("kurtosis(" + state.zs.pop() + ")");
            }
            
            NodeKurtosis* NodeKurtosis::clone_impl() const { return new NodeKurtosis(*this); }

            NodeKurtosis* NodeKurtosis::rnd_clone_impl() const { return new NodeKurtosis(); } 
        }
    }
}
