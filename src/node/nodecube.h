/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "node.h"

namespace FT{
	class NodeCube : public Node
    {
    	public:
    		  
    		NodeCube()
    		{
    			name = "cube";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 33;
    		}
    		
            /// Evaluates the node and updates the stack states.  
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(pow(stack.f.pop(),3));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "^3)");
            }
        protected:
            NodeCube* clone_impl() const override { return new NodeCube(*this); };  
    };
}	

#endif

