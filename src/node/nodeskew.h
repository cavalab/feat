/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SKEW
#define NODE_SKEW

#include "node.h"

namespace FT{
	class NodeSkew : public Node
    {
    	public:
    	
    		NodeSkew()
            {
                name = "skew";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['l'] = 1;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b, vector<vector<ArrayXd> > &stack_z)
            {
                ArrayXd tmp(stack_z.back().size());
                
                int x;
                
                for(x = 0; x < stack_z.back().size(); x++)
                    tmp(x) = skew(stack_z.back()[x]);
                    
                stack_z.pop_back();

                stack_f.push_back(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b, vector<string>& stack_z)
            {
                string x1 = stack_z.back(); stack_z.pop_back();
                stack_z.push_back("skew(" + x1 + ")");
            }
    };
}	

#endif
