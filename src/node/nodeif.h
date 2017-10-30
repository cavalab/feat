/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IF
#define NODE_IF

#include "node.h"

namespace FT{
	class NodeIf : public Node
    {
    	public:
    	   	
    		NodeIf()
    		{
    			name = "if";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 1;
    			complexity = 5;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
            	ArrayXb b = stack_b.back(); stack_b.pop_back();
                ArrayXd f = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(b.select(f,0));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string b = stack_b.back(); stack_b.pop_back();
                string f = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("(if-then-else(" + b + "," + f + "," + "0)");
            }
    };
}	

#endif
