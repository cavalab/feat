/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "node.h"

namespace FT{
	class NodeIfThenElse : public Node
    {
    	public:
    	
    		NodeIfThenElse()
    	    {
    			name = "ite";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 1;
    			complexity = 5;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);
            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                string f2 = stack.fs.pop();
                string f1 = stack.fs.pop();
                stack.fs.push("if-then-else(" + stack.bs.pop() + "," + f1 + "," + f2 + ")");
            }
        protected:
            NodeIfThenElse* clone_impl() const override { return new NodeIfThenElse(*this); };  
    };
#ifndef USE_CUDA
    void NodeIfThenElse::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd f2 = stack.f.pop();
        ArrayXd f1 = stack.f.pop();
        stack.f.push(stack.b.pop().select(f1,f2));
    }
#else
    void NodeIfThenElse::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_IfThenElse(stack.dev_f, stack.dev_b, stack.idx[otype], stack.idx['b'], stack.N);
    }
#endif

}	

#endif
