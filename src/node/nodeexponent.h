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
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
           		ArrayXd x2 = stack_f.back(); stack_f.pop_back();
                ArrayXd x1 = stack_f.back(); stack_f.pop_back();

                stack_f.push_back(limited(pow(W[1] * x1, W[0] * x2)));
            }
    
            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x2 = stack_f.back(); stack_f.pop_back();
                string x1 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("(" + x1 + ")^(" + x2 + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 3: // Weight for the base
                        return W[0] * stack_f[stack_f.size()-1] * limited(pow(W[1] * stack_f[stack_f.size() - 2], W[0] * stack_f[stack_f.size() - 1])) / W[1];
                    case 2: // Weight for the power
                        return limited(pow(W[1] * stack_f[stack_f.size() - 2], W[0] * stack_f[stack_f.size() - 1])) * limited(ln(W[1] * stack_f[stack_f.size() - 2])) * stack_f[stack_f.size() - 1];
                    case 1: // Base
                        return W[0] * stack_f[stack_f.size()-1] * limited(pow(W[1] * stack_f[stack_f.size() - 2], W[0] * stack_f[stack_f.size() - 1])) / stack_f[stack_f.size() - 2];
                    case 0: // Power
                    default:
                        return limited(pow(W[1] * stack_f[stack_f.size() - 2], W[0] * stack_f[stack_f.size() - 1])) * limited(ln(W[1] * stack_f[stack_f.size() - 2])) * W[0];
                } 
            }
        protected:
            NodeExponent* clone_impl() const override { return new NodeExponent(*this); };  
    };
}	

#endif
