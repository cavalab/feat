/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOGIT
#define NODE_LOGIT

#include "node.h"

namespace FT{
	class NodeLogit : public Node
    {
    	public:
    	
    		NodeLogit()
            {
                name = "logit";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);
            
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("1/(1+exp(-1*" + stack.fs.pop() + "))");
            }
        protected:
            NodeLogit* clone_impl() const override { return new NodeLogit(*this); };  
    };
#ifndef USE_CUDA
    void NodeLogit::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.f.push(1/(1+(limited(exp(-1*stack.f.pop())))));
    }
#else
    void NodeLogit::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_Logit(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif
}	

#endif
