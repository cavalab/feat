/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "node.h"

namespace FT{
	class NodeIfThenElse : public Node
    {
    	public:
    	
    		NodeIfThenElse()
    	    {
    			name = "ite";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 2;
    			complexity = 5;
    		}
    		
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
                ArrayXb b = stack_b.back(); stack_b.pop_back();
                ArrayXf f = stack_f.back(); stack_f.pop_back();
                ArrayXf f2 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(b.select(f,f2));
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string b = stack_b.back(); stack_b.pop_back();
                string f = stack_f.back(); stack_f.pop_back();
                string f2 = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("(if-then-else(" + b + "," + f + "," + f2 + ")");
            }
    };

}	

#endif