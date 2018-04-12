/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_XOR
#define NODE_XOR

#include "node.h"

namespace FT{
	class NodeXor : public Node
    {
    	public:
    	
    		NodeXor()
            {
                name = "xor";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);
            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x2 = stack.bs.pop();
                string x1 = stack.bs.pop();
                stack.bs.push("(" + x1 + " XOR " + x2 + ")");
            }
        protected:
            NodeXor* clone_impl() const override { return new NodeXor(*this); };  
    };
#ifndef USE_CUDA
    void NodeXor::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXb x2 = stack.b.pop();
        ArrayXb x1 = stack.b.pop();

        ArrayXb res = (x1 != x2).select(ArrayXb::Ones(x1.size()), ArrayXb::Zero(x1.size()));

        stack.b.push(res);
        
    }
#endif
}	

#endif
