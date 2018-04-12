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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
        		ArrayXd x = stack.f.pop();
        		ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), ArrayXd::Zero(x.size())); 
                stack.f.push(res);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("step("+ stack.fs.pop() +")");
            }
        protected:
            NodeStep* clone_impl() const override { return new NodeStep(*this); };  
    };
}	

#endif
