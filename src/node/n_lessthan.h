/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LESSTHAN
#define NODE_LESSTHAN

#include "node.h"

namespace FT{
	class NodeLessThan : public Node
    {
    	public:
    	
    		NodeLessThan()
       		{
    			name = "<";
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
                string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.bs.push("(" + x1 + "<" + x2 + ")");
            }
        protected:
            NodeLessThan* clone_impl() const override { return new NodeLessThan(*this); };  
    };
#ifndef USE_CUDA
    void NodeLessThan::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd x2 = stack.f.pop();
        ArrayXd x1 = stack.f.pop();
        stack.b.push(x1 < x2);
    }
#endif
}	

#endif
