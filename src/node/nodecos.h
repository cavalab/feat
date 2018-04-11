/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COS
#define NODE_COS

#include "node.h"

namespace FT{
	class NodeCos : public Node
    {
    	public:
    	  	
    		NodeCos()
    		{
    			name = "cos";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(cos(x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("cos(" + x + ")");
            }
        protected:
            NodeCos* clone_impl() const override { return new NodeCos(*this); };  
    };
}	

#endif
