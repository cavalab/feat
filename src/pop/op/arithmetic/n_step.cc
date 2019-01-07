/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_step.h"


namespace FT{

    namespace Pop{
        namespace Op{
            NodeStep::NodeStep()
            {
                name = "step";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeStep::evaluate(const Data& data, State& state)
            {
	            ArrayXf x = state.pop<float>();
	            ArrayXf res = (x > 0).select(ArrayXf::Ones(x.size()), ArrayXf::Zero(x.size())); 
                state.push<float>(res);
            }
            #else
            void NodeStep::evaluate(const Data& data, State& state)
            {
                GPU_Step(state.dev_f, state.idx[otype], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeStep::eval_eqn(State& state)
            {
                state.push<float>("step("+ state.popStr<float>() +")");
            }

            NodeStep* NodeStep::clone_impl() const { return new NodeStep(*this); }

            NodeStep* NodeStep::rnd_clone_impl() const { return new NodeStep(); }  
        }
    }
}
