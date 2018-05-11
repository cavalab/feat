/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COS
#define NODE_COS

#include "nodeDx.h"

namespace FT{
	class NodeCos : public NodeDx
    {
    	public:
    	  	
    		NodeCos(vector<double> W0 = vector<double>())
    		{
    			name = "cos";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(limited(cos(W[0] * stack.f.pop())));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("cos(" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return stack_f[stack_f.size()-1] * -sin(W[0] * stack_f[stack_f.size() - 1]);
                    case 0: // d/dx0
                    default:
                       return W[0] * -sin(W[0] * stack_f[stack_f.size() - 1]);
                } 
            }

        protected:
            NodeCos* clone_impl() const override { return new NodeCos(*this); };  
            NodeCos* rnd_clone_impl() const override { return new NodeCos(); };  
    };
}	

#endif
