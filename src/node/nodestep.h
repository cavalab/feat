/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_STEP
#define NODE_STEP

#include "node.h"

namespace FT{
	class NodeStep : public Node
    {
    	public:
    	
    		NodeStep()
            {
                name = "step";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
        		
        		ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), ArrayXd::Zero(x.size())); 
                stack_f.push_back(res);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("step("+ x +")");
            }
    };
}	

#endif
