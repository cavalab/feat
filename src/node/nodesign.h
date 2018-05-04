/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "nodeDx.h"

namespace FT{
	class NodeSign : public NodeDx
    {
    	public:
    	
    		NodeSign(vector<double> W0 = vector<double>())
            {
                name = "sign";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;

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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
        		ArrayXd x = stack.f.pop();
        	    ArrayXd ones = ArrayXd::Ones(x.size());

        		ArrayXd res = (W[0] * x > 0).select(ones, 
                                                    (x == 0).select(ArrayXd::Zero(x.size()), 
                                                                    -1*ones)); 
                stack.f.push(res);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("sign("+ stack.fs.pop() +")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                // Might want to experiment with using a perceptron update rule or estimating with some other function
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] / (2 * sqrt(W[0] * stack_f[stack_f.size()-1]));
                    case 0: // d/dx0
                    default:
                        return W[0] / (2 * sqrt(W[0] * stack_f[stack_f.size()-1]));
                } 
            }
        protected:
            NodeSign* clone_impl() const override { return new NodeSign(*this); };  
    };
}	

#endif
