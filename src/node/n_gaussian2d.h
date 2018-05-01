/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN2D
#define NODE_GAUSSIAN2D

#include "node.h"

namespace FT{
	class NodeGaussian2D : public Node
    {
    	public:
    	
    		NodeGaussian2D()
            {
                name = "2Dgaussian";
    			otype = 'f';
    			arity['f'] = 2;
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
        		string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.fs.push("gauss2d(" + x1 + "," + x2 + ")");
            }

        protected:
                NodeGaussian2D* clone_impl() const override { return new NodeGaussian2D(*this); };  
    };
	
#ifndef USE_CUDA
    void NodeGaussian2D::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd x2 = stack.f.pop();
        ArrayXd x1 = stack.f.pop();
        
        stack.f.push(limited(exp(-1*(pow((x1-x1.mean()), 2)/(2*variance(x1)) 
                          + pow((x2 - x2.mean()), 2)/variance(x2)))));
    }
#else
    void NodeGaussian2D::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        size_t idx = stack.idx['f'];
        ArrayXf x1 = ArrayXf::Map(stack.dev_f+(idx-1)*stack.N,stack.N);
        ArrayXf x2 = ArrayXf::Map(stack.dev_f+(idx-2)*stack.N,stack.N);

        GPU_Gaussian2D(stack.dev_f, x1.mean(), variance(x1), x2.mean(), variance(x2), 
                       stack.idx[otype], stack.N);
    }
#endif

}
#endif
