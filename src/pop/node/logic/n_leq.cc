/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_leq.h"
    	
namespace FT{

    namespace Pop{
        namespace NodeSpace{
            NodeLEQ::NodeLEQ()
            {
	            name = "<=";
	            otype = 'b';
	            arity['f'] = 2;
	            complexity = 2;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeLEQ::evaluate(const Data& data, Stacks& stack)
            {
              	ArrayXd x1 = stack.pop<double>();
                ArrayXd x2 = stack.pop<double>();
                stack.push<bool>(x1 <= x2);
            }

            /// Evaluates the node symbolically
            void NodeLEQ::eval_eqn(Stacks& stack)
            {
                stack.push<bool>("(" + stack.popStr<double>() + "<=" + stack.popStr<double>() + ")");
            }
            
            NodeLEQ* NodeLEQ::clone_impl() const { return new NodeLEQ(*this); }  

            NodeLEQ* NodeLEQ::rnd_clone_impl() const { return new NodeLEQ(); }
        }
    }
}
