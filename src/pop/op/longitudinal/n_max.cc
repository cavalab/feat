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

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeMax::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).maxCoeff();

                state.z.pop();
                
                state.push<float>(tmp);
                
            }
            #else
            void NodeMax::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).maxCoeff();

                state.z.pop();
                
                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);

                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeMax::eval_eqn(State& state)
            {
                state.push<float>("max(" + state.zs.pop() + ")");
            }
            
            NodeMax* NodeMax::clone_impl() const { return new NodeMax(*this); }

            NodeMax* NodeMax::rnd_clone_impl() const { return new NodeMax(); }
        }
    }
}
