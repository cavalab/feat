/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIGN
#define NODE_SIGN

#include "node.h"

namespace FT{
	class NodeSign : public Node
    {
    	public:
    	
    		NodeSign()
            {
                name = "sign";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);
            
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("sign("+ stack.fs.pop() +")");
            }
        protected:
            NodeSign* clone_impl() const override { return new NodeSign(*this); };  
    };
#ifndef USE_CUDA
    void NodeSign::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd x = stack.f.pop();
        ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), (x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
        stack.f.push(res);
    }
#endif
}	

#endif
