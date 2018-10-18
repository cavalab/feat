/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_square.h"
    	
namespace FT{

    namespace Pop{
        namespace NodeSpace{
            NodeSquare::NodeSquare(vector<double> W0)
            {
	            name = "^2";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 2;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeSquare::evaluate(const Data& data, Stacks& stack)
            {
                stack.push<double>(limited(pow(W[0]*stack.pop<double>(),2)));
            }

            /// Evaluates the node symbolically
            void NodeSquare::eval_eqn(Stacks& stack)
            {
                stack.push<double>("(" + stack.popStr<double>() + "^2)");
            }

            ArrayXd NodeSquare::getDerivative(Trace& stack, int loc)
            {
                ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
                switch (loc) {
                    case 1: // d/dw0
                        return 2 * pow(x, 2) * this->W[0];
                    case 0: // d/dx0
                    default:
                       return 2 * pow(this->W[0], 2) * x;
                } 
            }

            NodeSquare* NodeSquare::clone_impl() const { return new NodeSquare(*this); }

            NodeSquare* NodeSquare::rnd_clone_impl() const { return new NodeSquare(); }  
        }
    }
}
