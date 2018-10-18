/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/


#include "n_2dgaussian.h"
#include "../../../util/utils.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
            Node2dGaussian::Node2dGaussian(vector<double> W0)
            {
                name = "gaussian2d";
	            otype = 'f';
	            arity['f'] = 2;
	            complexity = 4;

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
            #ifndef USE_CUDA
            void Node2dGaussian::evaluate(const Data& data, Stacks& stack)
            {
                ArrayXd x1 = stack.pop<double>();
                ArrayXd x2 = stack.pop<double>();

                stack.push<double>(limited(exp(-1*(pow(W[0]*(x1-x1.mean()), 2)/(2*variance(x1)) 
                              + pow(W[1]*(x2 - x2.mean()), 2)/variance(x2)))));
            }
            #endif

            /// Evaluates the node symbolically
            void Node2dGaussian::eval_eqn(Stacks& stack)
            {
                stack.push<double>("gauss2d(" +stack.popStr<double>()+ "," +stack.popStr<double>()+ ")");
            }

            ArrayXd Node2dGaussian::getDerivative(Trace& stack, int loc) 
            {
                ArrayXd& x = stack.get<double>()[stack.size<double>()-1];

                switch (loc) {
                    case 1: // d/dw0
                        return -2 * W[0] * pow(x, 2) * exp(-pow(W[0] * x, 2));
                    case 0: // d/dx0
                    default:
                        return -2 * pow(W[0], 2) * x * exp(-pow(W[0] * x, 2));
                } 
            }
            
            Node2dGaussian* Node2dGaussian::clone_impl() const { return new Node2dGaussian(*this); }  

            Node2dGaussian* Node2dGaussian::rnd_clone_impl() const { return new Node2dGaussian(); }
        }
    }
}


