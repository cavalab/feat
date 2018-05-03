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
    	  	
    		NodeExponent()
    		{
    			name = "^";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 4;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
           		ArrayXd x2 = stack.f.pop();
                ArrayXd x1 = stack.f.pop();

                stack.f.push(limited(pow(this->W[1] * x1, this->W[0] * x2)));
            }
    
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.fs.push("(" + x1 + ")^(" + x2 + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                ArrayXd x1 = stack_f[stack_f.size() - 2];
                ArrayXd x2 = stack_f[stack_f.size() - 1];
                switch (loc) {
                    case 3: // Weight for the base
                        return limited(this->W[0] * x2 * pow(this->W[1] * x1, this->W[0] * x2) / this->W[1]);
                        // return this->W[0] * stack_f[stack_f.size()-1] * limited(pow(W[1] * stack_f[stack_f.size() - 2], this->W[0] * stack_f[stack_f.size() - 1])) / this->W[1];
                    case 2: // Weight for the power
                        return limited(pow(this->W[1] * x1, this->W[0] * x2) * limited(log(this->W[1] * x1)) * x2);
                        // return limited(pow(this->W[1] * stack_f[stack_f.size() - 2], this->W[0] * stack_f[stack_f.size() - 1])) * limited(log(this->W[1] * stack_f[stack_f.size() - 2])) * stack_f[stack_f.size() - 1];
                    case 1: // Base
                        return limited(this->W[0] * x2 * pow(this->W[1] * x1, this->W[0] *x2) / x2);
                        // return this->W[0] * stack_f[stack_f.size()-1] * limited(pow(W[1] * stack_f[stack_f.size() - 2], this->W[0] * stack_f[stack_f.size() - 1])) / stack_f[stack_f.size() - 2];
                    case 0: // Power
                    default:
                        return limited(pow(this->W[1] * x1, this->W[0] * x2) * limited(log(this->W[1] * x1)) * this->W[0]);
                        //return limited(pow(W[1] * stack_f[stack_f.size() - 2], W[0] * stack_f[stack_f.size() - 1])) * limited(log(this->W[1] * stack_f[stack_f.size() - 2])) * this->W[0];
                } 
            }
        protected:
            NodeExponent* clone_impl() const override { return new NodeExponent(*this); };  
    };
}	

#endif
