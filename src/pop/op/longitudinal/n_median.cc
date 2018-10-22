/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_median.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeMedian::NodeMedian()
            {
                name = "median";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeMedian::evaluate(const Data& data, Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                
                int x;
                
                for(x = 0; x < stack.z.top().first.size(); x++)
                    tmp(x) = median(limited(stack.z.top().first[x]));
                    
                stack.z.pop();

                stack.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeMedian::eval_eqn(Stacks& stack)
            {
                stack.push<double>("median(" + stack.zs.pop() + ")");
            }
            
            NodeMedian* NodeMedian::clone_impl() const { return new NodeMedian(*this); }

            NodeMedian* NodeMedian::rnd_clone_impl() const { return new NodeMedian(); } 
        }
    }
}
