/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SUBTRACT
#define NODE_SUBTRACT

#include "nodeDx.h"

namespace FT{
	class NodeSubtract : public NodeDx
    {
    	public:
    	
    		NodeSubtract()
    		{
    			name = "-";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 1;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(r());
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(limited(W[1]*stack.f.pop() - W[0]*stack.f.pop()));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x1 = stack.fs.pop();
                string x2 = stack.fs.pop();
                stack.fs.push("(" + x1 + "-" + x2 + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 3: 
                        return stack_f[stack_f.size()-2];
                    case 2: 
                        return -stack_f[stack_f.size()-1];
                    case 1:
                        return this->W[1] * ArrayXd::Ones(stack_f[stack_f.size()-1].size());
                    case 0:
                    default:
                       return -this->W[0] * ArrayXd::Ones(stack_f[stack_f.size()-2].size());
                } 
            }

        protected:
            NodeSubtract* clone_impl() const override { return new NodeSubtract(*this); };  
    };
}	

#endif
