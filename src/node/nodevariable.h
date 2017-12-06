/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_VARIABLE
#define NODE_VARIABLE

#include "node.h"

namespace FT{
	class NodeVariable : public Node
	{
		public:
			size_t loc;             ///< column location in X, for x types
			
			NodeVariable(const size_t& l, char ntype = 'f', std::string n="")
			{
                if (n.empty())
    			    name = "x_" + std::to_string(l);
                else
                    name = n;
    			otype = ntype;
    			arity['f'] = 0;
    			arity['b'] = 0;
    			complexity = 1;
    			loc = l;
    		}
    		
    		/// Evaluates the node and updates the stack states. 		
			void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
		    {
	    		if (otype == 'b')
	                stack_b.push_back(X.row(loc).cast<bool>());
	            else
	                stack_f.push_back(X.row(loc));
		    }

		    /// Evaluates the node symbolically
		    void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
		    {
	    		if (otype == 'b')
	                stack_b.push_back(name);
	            else
	                stack_f.push_back(name);
		    }
	};
}

#endif
