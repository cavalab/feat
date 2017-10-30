/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GEQ
#define NODE_GEQ

#include "node.h"

namespace FT{
	class NodeGEQ : public Node
    {
    	public:
    	
   		    NodeGEQ()
    		{
    			name = ">=";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
            	ArrayXd x2 = stack_f.back(); stack_f.pop_back();
                ArrayXd x1 = stack_f.back(); stack_f.pop_back();
                stack_b.push_back(x1 >= x2);
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string x2 = stack_f.back(); stack_f.pop_back();
                string x1 = stack_f.back(); stack_f.pop_back();
                stack_b.push_back("(" + x1 + ">=" + x2 + ")");
            }
    };
}	

#endif
