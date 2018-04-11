/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOG
#define NODE_LOG

#include "nodeDx.h"

namespace FT{
	class NodeLog : public NodeDx
    {
    	public:
    	
    		NodeLog()
       		{
    			name = "log";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}

            /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
           		ArrayXd x = stack_f.back(); stack_f.pop_back();                    
                stack_f.push_back( (abs(x) > NEAR_ZERO).select(log(abs(W[0] * x)),MIN_DBL) );
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("log(" + x + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return 1/(W[0]]);
                    case 0: // d/dx0
                    default:
                       return 1/(stack_f[stack_f.size()-1]);
                } 
            }
        protected:
            NodeLog* clone_impl() const override { return new NodeLog(*this); };  
    };
}	

#endif
