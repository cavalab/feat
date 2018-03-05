/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LONGITUDINAL
#define NODE_LONGITUDINAL

#include "node.h"

namespace FT{
	class NodeLongitudinal : public Node
	{
		public:
			size_t loc;             ///< column location in X, for x types
			
			NodeLongitudinal(const size_t& l, std::string n = "")
			{
			    if(n.empty())
                    name = "l_" + std::to_string(l);
                else
                    name = n;
                    
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['l'] = 1;
    			complexity = 1;
    			loc = l;
    		}
    		
    		/// Evaluates the node and updates the stack states. 		
			void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b, vector<vector<ArrayXd> > &stack_z)
		    {
		        stack_z.push_back(Z[loc]);
		    }

		    /// Evaluates the node symbolically
		    void eval_eqn(vector<string>& stack_f, vector<string>& stack_b, vector<string>& stack_z)
		    {
		        stack_z.push_back(name);
		    }
	};
}

#endif
