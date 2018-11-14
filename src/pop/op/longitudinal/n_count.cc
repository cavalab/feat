/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_count.h"
    	
namespace FT{


    namespace Pop{
        namespace Op{

            NodeCount::NodeCount()
            {
                name = "count";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            #ifndef USE_CUDA 
            /// Evaluates the node and updates the state states. 
            void NodeCount::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)
                    tmp(x) = limited(state.z.top().first[x]).cols();
                  
                state.z.pop();
                
                state.push<double>(tmp);
                
            }
            #else
            void NodeCount::evaluate(const Data& data, State& state)
            {
                
                for(unsigned x = 0; x < state.z.top().first.size(); x++)
                    state.f.row(state.idx['f']) = limited(state.z.top().first[x]).cols();
                  
                state.z.pop();
                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeCount::eval_eqn(State& state)
            {
                state.push<double>("count(" + state.zs.pop() + ")");
            }
            
            NodeCount*NodeCount::clone_impl() const { return new NodeCount(*this); }
             
            NodeCount* NodeCount::rnd_clone_impl() const { return new NodeCount(); }
        }
    }
}
