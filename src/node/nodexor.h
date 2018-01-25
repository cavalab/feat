/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_XOR
#define NODE_XOR

#include "node.h"

namespace FT{
	class NodeXor : public Node
    {
    	public:
    	
    		NodeXor()
            {
                name = "xor";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXb x2 = stack_b.back(); stack_b.pop_back();
                ArrayXb x1 = stack_b.back(); stack_b.pop_back();

                ArrayXb res = (x1 != x2).select(ArrayXb::Ones(x1.size()), ArrayXb::Zero(x1.size()));

                stack_b.push_back(res);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x2 = stack_b.back(); stack_b.pop_back();
                string x1 = stack_b.back(); stack_b.pop_back();
                stack_b.push_back("(" + x1 + " XOR " + x2 + ")");
            }
    };
}	

#endif
