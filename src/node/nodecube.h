/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "node.h"

namespace FT{
	class NodeCube : public Node
    {
    	public:
    		  
    		NodeCube()
    		{
    			name = "cube";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 33;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states.  
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(pow(W[0] * x,3));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("(" + x + "^3)");
            }

            ArrayXd getDerivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return 3 * pow(stack_f[stack_f.size()-1], 3) * pow(W[0], 2);
                    case 0: // d/dx0
                    default:
                       return 3 * pow(W[0], 3) * pow(stack_f[stack_f.size()-1], 2);
                } 
            }
    };
}	

#endif

