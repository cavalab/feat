/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_AND
#define NODE_AND

#include "node.h"

namespace FT{
	class NodeAnd : public Node
    {
    	public:
    	
    		NodeAnd()
       		{
    			name = "and";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);
            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.bs.pop() + " AND " + stack.bs.pop() + ")");
            }
        protected:
            virtual NodeAnd* clone_impl() const override { return new NodeAnd(*this); };  
    };
#ifndef USE_CUDA	
    void NodeAnd::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.b.push_back(stack.b.pop() && stack.b.pop());

    }
#endif
}
#endif
