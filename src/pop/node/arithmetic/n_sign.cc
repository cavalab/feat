/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sign.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
            NodeSign::NodeSign()
            {
                name = "sign";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 1;

            }

            /// Evaluates the node and updates the stack states. 
            void NodeSign::evaluate(const Data& data, Stacks& stack)
            {
	            ArrayXd x = stack.pop<double>();
                ArrayXd ones = ArrayXd::Ones(x.size());

	            ArrayXd res = ( x > 0).select(ones, 
                                                    (x == 0).select(ArrayXd::Zero(x.size()), 
                                                                    -1*ones)); 
                stack.push<double>(res);
            }

            /// Evaluates the node symbolically
            void NodeSign::eval_eqn(Stacks& stack)
            {
                stack.push<double>("sign("+ stack.popStr<double>() +")");
            }

            
            NodeSign* NodeSign::clone_impl() const { return new NodeSign(*this); }

            NodeSign* NodeSign::rnd_clone_impl() const { return new NodeSign(); }  
        }
    }
}
