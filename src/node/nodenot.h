/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_NOT
#define NODE_NOT

#include "node.h"

namespace FT{
	class NodeNot : public Node
    {
    	public:
    	
    		NodeNot()
       		{
    			name = "not";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 1;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
            	ArrayXb x = stack_b.back(); stack_b.pop_back();
                stack_b.push_back(!x);
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string x = stack_b.back(); stack_b.pop_back();
                stack_b.push_back("NOT(" + x + ")");
            }
    };
    
}	

#endif
