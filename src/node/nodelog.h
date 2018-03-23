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
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
			{
           		ArrayXd x = stack.f.pop();
                stack.f.push( (abs(x) > NEAR_ZERO).select(log(abs(x)),MIN_DBL) );
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("log(" + stack.fs.pop() + ")");
            }
    };
}	

#endif
