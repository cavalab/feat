/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_2DGAUSSIAN
#define NODE_2DGAUSSIAN

#include "node.h"

namespace FT{
	class Node2dGaussian : public Node
    {
    	public:
    	
    		Node2dGaussian()
            {
                name = "2dgaussian";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
        		ArrayXd x1 = stack.f.pop();
                ArrayXd x2 = stack.f.pop();
                
                stack.f.push(limited(exp(-1*(pow((x1-x1.mean()), 2)/(2*variance(x1)) 
                                  + pow((x2 - x2.mean()), 2)/variance(x2)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("gauss2d(" + stack.fs.pop() + "," + stack.fs.pop() + ")");
            }

        protected:
                Node2dGaussian* clone_impl() const override { return new Node2dGaussian(*this); };  
    };
}	

#endif
