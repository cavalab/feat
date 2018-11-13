/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_and.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeAnd::NodeAnd()
            {
	            name = "and";
	            otype = 'b';
	            arity['b'] = 2;
	            complexity = 2;
            }

            /// Evaluates the node and updates the state states. 
            void NodeAnd::evaluate(const Data& data, State& state)
            {
                state.push<bool>(state.pop<bool>() && state.pop<bool>());

            }

            /// Evaluates the node symbolically
            void NodeAnd::eval_eqn(State& state)
            {
                state.push<bool>("(" + state.popStr<bool>() + " AND " + state.popStr<bool>() + ")");
            }
            
            NodeAnd* NodeAnd::clone_impl() const { return new NodeAnd(*this); }
              
            NodeAnd* NodeAnd::rnd_clone_impl() const { return new NodeAnd(); } 
        }
    }
}
