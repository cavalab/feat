/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_time.h"
    
namespace FT{

    namespace Pop{
        namespace Op{	
            NodeTime::NodeTime()
            {
                name = "time";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            /// Evaluates the node and updates the state states. 
            void NodeTime::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x])[0];
                    
                state.z.pop();

                state.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeTime::eval_eqn(State& state)
            {
                state.push<double>("time(" + state.zs.pop() + ")");
            }
            
            NodeTime* NodeTime::clone_impl() const { return new NodeTime(*this); }

            NodeTime* NodeTime::rnd_clone_impl() const { return new NodeTime(); }
        }
    }
}

