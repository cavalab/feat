/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "node.h"

namespace FT{
	class NodeSign : public Node
    {
    	public:
    	
    		NodeSign()
            {
                name = "sign";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
        		
        		ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), (x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                stack_f.push_back(res);
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("sign("+ x +")");
            }
    };
}	

#endif
