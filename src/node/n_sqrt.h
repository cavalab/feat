/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQRT
#define NODE_SQRT

#include "node.h"

namespace FT{
	class NodeSqrt : public Node
    {
    	public:
    	
    		NodeSqrt()
    		{
    			std::cerr << "error in noderoot.h : invalid constructor called";
				throw;
    		}
    	
    		NodeSqrt(string n)
    		{
    			name = n;
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f,
                    vector<ArrayXb>& stack_b);

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("sqrt(|" + x + "|)");
            	}
            }
    };
#ifndef USE_CUDA
    void NodeSqrt::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
            vector<ArrayXb>& stack_b)
    {
        if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
        {
            ArrayXd x = stack_f.back(); stack_f.pop_back();
            stack_f.push_back(sqrt(abs(x)));
        }
    }	
#endif

}

#endif
