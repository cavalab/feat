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

            /// Evaluates the node and updates the state states. 
            void NodeMode::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).mean();
                    
                state.z.pop();

                state.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeMode::eval_eqn(State& state)
            {
                state.push<double>("mean(" + state.zs.pop() + ")");
            }
            
            NodeMode* NodeMode::clone_impl() const { return new NodeMode(*this); }

            NodeMode* NodeMode::rnd_clone_impl() const { return new NodeMode(); }  
        }
    }
}
