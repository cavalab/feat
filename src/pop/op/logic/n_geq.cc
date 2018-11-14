/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_geq.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeGEQ::NodeGEQ()
            {
	            name = ">=";
	            otype = 'b';
	            arity['f'] = 2;
	            complexity = 2;
            }

            /// Evaluates the node and updates the state states. 
            void NodeGEQ::evaluate(const Data& data, State& state)
            {
	            ArrayXd x1 = state.pop<double>();
                ArrayXd x2 = state.pop<double>();
                state.push<bool>(x1 >= x2);
            }
            #else
            void NodeGEQ::evaluate(const Data& data, State& state)
            {
                GPU_GEQ(state.dev_f, state.dev_b, state.idx['f'], state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeGEQ::eval_eqn(State& state)
            {
                state.push<bool>("(" + state.popStr<double>() + ">=" + state.popStr<double>() + ")");
            }
            
            NodeGEQ* NodeGEQ::clone_impl() const { return new NodeGEQ(*this); }
              
            NodeGEQ* NodeGEQ::rnd_clone_impl() const { return new NodeGEQ(); } 
        }
    }
}
