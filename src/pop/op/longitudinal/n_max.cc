/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_max.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeMax::NodeMax()
            {
                name = "max";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            /// Evaluates the node and updates the state states. 
            void NodeMax::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).maxCoeff();

                state.z.pop();
                
                state.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeMax::eval_eqn(State& state)
            {
                state.push<double>("max(" + state.zs.pop() + ")");
            }
            
            NodeMax* NodeMax::clone_impl() const { return new NodeMax(*this); }

            NodeMax* NodeMax::rnd_clone_impl() const { return new NodeMax(); }
        }
    }
}
