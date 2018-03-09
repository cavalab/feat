/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_ROOT
#define NODE_ROOT

#include "node.h"

namespace FT{
	class NodeRoot : public Node
    {
    	public:
    	
    		NodeRoot()
    		{
    			std::cerr << "error in noderoot.h : invalid constructor called";
				throw;
    		}
    	
    		NodeRoot(string n)
    		{
    			name = n;
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
            	if (stack.f.size() >= arity['f'] && stack.b.size() >= arity['b'])
            	{
            		ArrayXd x = stack.f.pop();
                    stack.f.push(sqrt(abs(x)));
            	}
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
            	if (stack.f.size() >= arity['f'] && stack.b.size() >= arity['b'])
            	{
            		string x = stack.fs.pop();
                    stack.fs.push("sqrt(|" + x + "|)");
            	}
            }
    };
}	

#endif
