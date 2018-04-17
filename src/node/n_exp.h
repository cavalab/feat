/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EXP
#define NODE_EXP

#include "node.h"

namespace FT{
	class NodeExp : public Node
    {
    	public:
   	
    		NodeExp()
    		{
    			name = "exp";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)   ;

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("exp(" + stack.fs.pop() + ")");
            }

        protected:
            NodeExp* clone_impl() const override { return new NodeExp(*this); };  
    };
#ifndef USE_CUDA
    void NodeExp::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)   
    {
        stack.f.push(limited(exp(stack.f.pop())));
    }
#else
    void NodeExp::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_Exp(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif
}	

#endif
