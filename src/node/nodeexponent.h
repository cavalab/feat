/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EXPONENT
#define NODE_EXPONENT

#include "nodeDx.h"

namespace FT{
	class NodeExponent : public NodeDx
    {
    	public:
    	  	
    		NodeExponent(vector<double> W0 = vector<double>())
    		{
    			name = "^";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
           		/* ArrayXd x1 = stack.f.pop(); */
                /* ArrayXd x2 = stack.f.pop(); */

                stack.f.push(limited(pow(this->W[0] * stack.f.pop(), this->W[1] * stack.f.pop())));
            }
    
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + ")^(" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                ArrayXd x1 = stack_f[stack_f.size() - 1];
                ArrayXd x2 = stack_f[stack_f.size() - 2];
                switch (loc) {
                    case 3: // Weight for the power
                        return limited(pow(this->W[0] * x1, this->W[1] * x2) * limited(log(this->W[0] * x1)) * x2);
                    case 2: // Weight for the base
                        return limited(this->W[1] * x2 * pow(this->W[0] * x1, this->W[1] * x2) / this->W[0]);
                    case 1: // Power
                        return limited(this->W[1]*pow(this->W[0] * x1, this->W[1] * x2) * limited(log(this->W[0] * x1)));
                    case 0: // Base
                    default:
                        return limited(this->W[1] * x2 * pow(this->W[0] * x1, this->W[1] * x2) / x1);
                } 
            }
        protected:
            NodeExponent* clone_impl() const override { return new NodeExponent(*this); };  
    };
}	

#endif
