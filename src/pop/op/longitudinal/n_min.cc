/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_min.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeMin::NodeMin()
            {
                name = "min";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeMin::evaluate(const Data& data, State& state)
            {
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).minCoeff();
                    
                state.z.pop();

                state.push<float>(tmp);
                
            }
            #else
            void NodeMin::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).minCoeff();
                    
                state.z.pop();

                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);

                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeMin::eval_eqn(State& state)
            {
                state.push<float>("min(" + state.zs.pop() + ")");
            }
            
            NodeMin* NodeMin::clone_impl() const { return new NodeMin(*this); }

            NodeMin* NodeMin::rnd_clone_impl() const { return new NodeMin(); } 
        }
    }
}
