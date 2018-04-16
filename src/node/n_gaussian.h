/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "node.h"

namespace FT{
	class NodeGaussian : public Node
    {
    	public:
    	
    		NodeGaussian()
            {
                name = "gaussian";
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
        		string x = stack.fs.pop();
                stack.fs.push("exp(-(" + stack.fs.pop() + " ^ 2))");
            }
        protected:
            NodeGaussian* clone_impl() const override { return new NodeGaussian(*this); };  
    };
#ifndef USE_CUDA
    void NodeGaussian::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        stack.f.push(limited(exp(-1*limited(pow(stack.f.pop(), 2)))));
    }
#else
    void NodeGaussian::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_Gaussian(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif
}	

#endif
