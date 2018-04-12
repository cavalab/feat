/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GEQ
#define NODE_GEQ

#include "node.h"

namespace FT{
	class NodeGEQ : public Node
    {
    	public:
    	
   		    NodeGEQ()
    		{
    			name = ">=";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.fs.pop() + ">=" + stack.fs.pop() + ")");
            }
        protected:
            NodeGEQ* clone_impl() const override { return new NodeGEQ(*this); };  
    };
#ifndef USE_CUDA
    void NodeGEQ::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.b.push(stack.f.pop() >= stack.f.pop());
    }
#endif
}	

#endif
