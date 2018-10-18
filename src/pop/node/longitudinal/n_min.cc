/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_min.h"
    	
namespace FT{

    namespace Pop{
        namespace NodeSpace{
            NodeMin::NodeMin()
            {
                name = "min";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 1;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeMin::evaluate(const Data& data, Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                
                int x;
                
                for(x = 0; x < stack.z.top().first.size(); x++)
                    tmp(x) = limited(stack.z.top().first[x]).minCoeff();
                    
                stack.z.pop();

                stack.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeMin::eval_eqn(Stacks& stack)
            {
                stack.push<double>("min(" + stack.zs.pop() + ")");
            }
            
            NodeMin* NodeMin::clone_impl() const { return new NodeMin(*this); }

            NodeMin* NodeMin::rnd_clone_impl() const { return new NodeMin(); } 
        }
    }
}
