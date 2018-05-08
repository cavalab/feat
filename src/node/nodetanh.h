/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_TANH
#define NODE_TANH

#include "node.h"

namespace FT{
	class NodeTanh : public Node
    {
    	public:
    	
    		NodeTanh()
            {
                name = "tanh";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
			{
                stack.f.push(limited(tanh(stack.f.pop())));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("tanh(" + stack.fs.pop() + ")");
            }
        protected:
            NodeTanh* clone_impl() const override { return new NodeTanh(*this); };  
            NodeTanh* rnd_clone_impl() const override { return new NodeTanh(); };  
    };
}	

#endif
