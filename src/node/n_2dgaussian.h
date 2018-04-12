/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_2DGAUSSIAN
#define NODE_2DGAUSSIAN

#include "node.h"

namespace FT{
	class Node2DGaussian : public Node
    {
    	public:
    	
    		Node2DGaussian()
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
			              Stacks& stack)   ;

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.fs.push("gauss2d(" + x1 + "," + x2 + ")");
            }

        protected:
                Node2dGaussian* clone_impl() const override { return new Node2dGaussian(*this); };  
    };
	
#ifndef USE_CUDA
    void Node2DGaussian::void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd x2 = stack.f.pop();
        ArrayXd x1 = stack.f.pop();
        
        stack.f.push(limited(exp(-1*(pow((x1-x1.mean()), 2)/(2*variance(x1)) 
                          + pow((x2 - x2.mean()), 2)/variance(x2)))));
    }
#endif

}
#endif
