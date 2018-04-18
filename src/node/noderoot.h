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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                    stack.f.push(limited(sqrt(abs(stack.f.pop()))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                    stack.fs.push("sqrt(|" + stack.fs.pop() + "|)");
            }
        protected:
            NodeRoot* clone_impl() const override { return new NodeRoot(*this); };  
    };
}	

#endif
