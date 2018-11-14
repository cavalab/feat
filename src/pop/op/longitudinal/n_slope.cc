/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_slope.h"
#include "../../../util/utils.h"    	

namespace FT{

    namespace Pop{
        namespace Op{
            NodeSlope::NodeSlope()
            {
                name = "slope";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 4;
            }
            
            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeSlope::evaluate(const Data& data, State& state)
            {
                ArrayXd tmp(state.z.top().first.size());
                
                for(int x = 0; x < state.z.top().first.size(); x++)                    
                    tmp(x) = slope(limited(state.z.top().first[x]), limited(state.z.top().second[x]));
                    
                state.z.pop();

                state.push<double>(tmp);
                
            }
            #else
            void NodeSlope::evaluate(const Data& data, State& state)
            {
                
                int x;
                
                for(x = 0; x < state.z.top().first.size(); x++)                    
                    state.f.row(state.idx['f']) = slope(state.z.top().first[x], state.z.top().second[x]);
                    
                state.z.pop();

                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSlope::eval_eqn(State& state)
            {
                state.push<double>("slope(" + state.zs.pop() + ")");
            }
            
            double NodeSlope::slope(const ArrayXd& x, const ArrayXd& y)
            {
                double varx = variance(x);
                if (varx > NEAR_ZERO)
                    return covariance(x, y)/varx;
                else
                    return 0;
            }

            NodeSlope* NodeSlope::clone_impl() const { return new NodeSlope(*this); }

            NodeSlope* NodeSlope::rnd_clone_impl() const { return new NodeSlope(); } 
        }
    }
}
