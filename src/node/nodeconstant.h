/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CONSTANT
#define NODE_CONSTANT

#include "node.h"

namespace FT{
	class NodeConstant : public Node
    {
    	public:
    		
    		double value;           ///< value, for k and x types
    		
    		NodeConstant()
    		{
    			std::cerr << "error in node.h : invalid constructor called";
				throw;
    		}
    		
    		NodeConstant(std::string n, bool& v) : name(n), 
    												otype('b'),
    												arity['f'](0), 
    												arity['b'](0),
    												complexity(1), 
    												value(v) {}
    		
    		
    		NodeConstant(std::string n, const double& v) : name(n), 
    												otype('f'),
    												arity['f'](0), 
    												arity['b'](0),
    												complexity(1), 
    												value(v) {}
    		
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXi>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		if (otype == 'b')
                        stack_b.push_back(ArrayXi::Constant(X.cols(),int(value)));
                    else 	
                        stack_f.push_back(ArrayXd::Constant(X.cols(),value));
            	}
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		if (otype == 'b')
                        stack_b.push_back(std::to_string(value));
                    else 	
                        stack_f.push_back(std::to_string(value));
            	}
            }
    		
    };
}	
