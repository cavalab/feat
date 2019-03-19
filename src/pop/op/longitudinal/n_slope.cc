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
                ArrayXf tmp(state.z.top().first.size());
                
                for(int x = 0; x < state.z.top().first.size(); x++)                    
                {
                    /* cout << "x: " << x << "\n"; */
                    /* cout << "value: " << state.z.top().first[x].transpose() << "\n"; */
                    /* cout << "date: " << state.z.top().second[x].transpose() << "\n"; */
                    tmp(x) = slope(limited(state.z.top().second[x]), 
                                   limited(state.z.top().first[x]));
                    /* cout << "slope: " << tmp(x) << "\n"; */
                }
                    
                state.z.pop();

                state.push<float>(tmp);
                
            }
            #else
            void NodeSlope::evaluate(const Data& data, State& state)
            {
                
                ArrayXf tmp(state.z.top().first.size());
                
                for(int x = 0; x < state.z.top().first.size(); x++)                    
                    tmp(x) = slope(limited(state.z.top().first[x]), limited(state.z.top().second[x]));
                    
                state.z.pop();

                GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);

                
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSlope::eval_eqn(State& state)
            {
                state.push<float>("slope(" + state.zs.pop() + ")");
            }
            
            float NodeSlope::slope(const ArrayXf& x, const ArrayXf& y)
            {
                float varx = variance(x);
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
