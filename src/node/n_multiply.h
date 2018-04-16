/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MULTIPLY
#define NODE_MULTIPLY

#include "node.h"

namespace FT{
	class NodeMultiply : public Node
    {
    	public:
    	
    		NodeMultiply()
       		{
    			name = "*";
    			otype = 'f';
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
            	stack.fs.push("(" + stack.fs.pop() + "*" + stack.fs.pop() + ")");
            }
        protected:
            NodeMultiply* clone_impl() const override { return new NodeMultiply(*this); };  
    };
#ifndef USE_CUDA
    void NodeMultiply::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.f.push(limited(stack.f.pop() * stack.f.pop()));
    }
#else
    void NodeMultiply::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_Multiply(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif
}	

#endif
