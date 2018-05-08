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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(limited(cos(stack.f.pop())));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("cos(" + stack.fs.pop() + ")");
            }
        protected:
            NodeCos* clone_impl() const override { return new NodeCos(*this); };  
            NodeCos* rnd_clone_impl() const override { return new NodeCos(); };  
    };
}	

#endif
