/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MULTIPLY
#define NODE_MULTIPLY

#include "nodeDx.h"

namespace FT{
	class NodeMultiply : public NodeDx
    {
    	public:
    	
    		NodeMultiply()
       		{
    			name = "*";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(r());
                }
    		}

            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(limited(W[1]*stack.f.pop() * W[0]*stack.f.pop()));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
            	stack.fs.push("(" + stack.fs.pop() + "*" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 3: 
                        return stack_f[stack_f.size()-1] * this->W[0] * stack_f[stack_f.size()-2];
                    case 2: 
                        return stack_f[stack_f.size()-1] * this->W[1] * stack_f[stack_f.size()-2];
                    case 1:
                        return this->W[0] * this->W[1] * stack_f[stack_f.size() - 1];
                    case 0:
                    default:
                        return this->W[1] * this->W[0] * stack_f[stack_f.size() - 2];
                } 
            }

        protected:
            NodeMultiply* clone_impl() const override { return new NodeMultiply(*this); };  
    };
}	

#endif
