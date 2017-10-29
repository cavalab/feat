/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOG
#define NODE_LOG

#include "node.h"

namespace FT{
	class NodeLog : public Node
    {
    	public:
    	
    		NodeLog()
       		{
    			name = "log";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;
    		}

            /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
           		ArrayXd x = stack_f.back(); stack_f.pop_back();                    
                stack_f.push_back( (abs(x) > NEAR_ZERO).select(log(abs(x)),MIN_DBL) );
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("log(" + x + ")");
            }
    };
}	

#endif
