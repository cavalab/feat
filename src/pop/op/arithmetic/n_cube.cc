/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_cube.h"

namespace FT{

    namespace Pop{
        namespace Op{    		  
            NodeCube::NodeCube(vector<double> W0)
            {
		        name = "cube";
		        otype = 'f';
		        arity['f'] = 1;
		        arity['b'] = 0;
		        complexity = 3;

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
            void NodeCube::evaluate(const Data& data, Stacks& stack)
            {
                stack.push<double>(limited(pow(this->W[0] * stack.pop<double>(),3)));
            }

            /// Evaluates the node symbolically
            void NodeCube::eval_eqn(Stacks& stack)
            {
                stack.push<double>("(" + stack.popStr<double>() + "^3)");
            }

            ArrayXd NodeCube::getDerivative(Trace& stack, int loc)
            {
                ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
                
                switch (loc) {
                    case 1: // d/dw0
                        return 3 * pow(x, 3) * pow(this->W[0], 2);
                    case 0: // d/dx0
                    default:
                       return 3 * pow(this->W[0], 3) * pow(x, 2);
                } 
            }
            
            NodeCube* NodeCube::clone_impl() const { return new NodeCube(*this); }
              
            NodeCube* NodeCube::rnd_clone_impl() const { return new NodeCube(); } 
        }
    }
}
