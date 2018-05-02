/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COUNT
#define NODE_COUNT

#include "node.h"

namespace FT{
	class NodeCount : public Node
    {
    	public:
    	
    		NodeCount()
            {
                name = "count";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['z'] = 1;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                int x;
                
                for(x = 0; x < stack.z.top().first.size(); x++)
                    tmp(x) = limited(stack.z.top().first[x]).cols();
                  
                stack.z.pop();
                
                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("count(" + stack.zs.pop() + ")");
            }
        protected:
            NodeCount* clone_impl() const override { return new NodeCount(*this); }; 
    };
}	

#endif
