/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LEQ
#define NODE_LEQ

#include "node.h"

namespace FT{
	class NodeLEQ : public Node
    {
    	public:
    	
    		NodeLEQ()
    		{
    			name = "<=";
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
                stack.bs.push("(" + stack.fs.pop() + "<=" + stack.fs.pop() + ")");
            }

        protected:
            NodeLEQ* clone_impl() const override { return new NodeLEQ(*this); };  
    };
#ifndef USE_CUDA
    void NodeLEQ::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.b.push(stack.f.pop() <= stack.f.pop());
    }
#endif
}	

#endif
