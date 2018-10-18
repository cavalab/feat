/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_or.h"
    	
namespace FT{

    
    namespace Pop{
        namespace NodeSpace{
            NodeOr::NodeOr()
            {
                name = "or";
	            otype = 'b';
	            arity['b'] = 2;
	            complexity = 2;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeOr::evaluate(const Data& data, Stacks& stack)
            {
                stack.push<bool>(stack.pop<bool>() || stack.pop<bool>());

            }

            /// Evaluates the node symbolically
            void NodeOr::eval_eqn(Stacks& stack)
            {
                stack.push<bool>("(" + stack.popStr<bool>() + " OR " + stack.popStr<bool>() + ")");
            }
            
            NodeOr* NodeOr::clone_impl() const { return new NodeOr(*this); }

            NodeOr* NodeOr::rnd_clone_impl() const { return new NodeOr(); }  
        }
    }

}
