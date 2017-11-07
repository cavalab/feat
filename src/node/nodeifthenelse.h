/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "node.h"

namespace FT{
	class NodeIfThenElse : public Node
    {
    	public:
    	
    		NodeIfThenElse()
    	    {
    			name = "ite";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 1;
    			complexity = 5;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
                ArrayXb b = stack_b.back(); stack_b.pop_back();
                ArrayXd f2 = stack_f.back(); stack_f.pop_back();
                ArrayXd f1 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(b.select(f1,f2));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string b = stack_b.back(); stack_b.pop_back();
                string f2 = stack_f.back(); stack_f.pop_back();
                string f1 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("(if-then-else(" + b + "," + f1 + "," + f2 + ")");
            }
    };

}	

#endif
