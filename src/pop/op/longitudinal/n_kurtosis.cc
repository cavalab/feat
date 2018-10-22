/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_kurtosis.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace Op{

            NodeKurtosis::NodeKurtosis()
            {
                name = "kurtosis";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeKurtosis::evaluate(const Data& data, Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                
                int x;
                
                for(x = 0; x < stack.z.top().first.size(); x++)
                    tmp(x) = kurtosis(limited(stack.z.top().first[x]));
                    
                stack.z.pop();
                stack.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeKurtosis::eval_eqn(Stacks& stack)
            {
                stack.push<double>("kurtosis(" + stack.zs.pop() + ")");
            }
            
            NodeKurtosis* NodeKurtosis::clone_impl() const { return new NodeKurtosis(*this); }

            NodeKurtosis* NodeKurtosis::rnd_clone_impl() const { return new NodeKurtosis(); } 
        }
    }
}
