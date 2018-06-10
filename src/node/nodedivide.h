/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#include "nodeDx.h"

namespace FT{
	class NodeDivide : public NodeDx
    {
    	public:
    	  	
    		NodeDivide(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            // Might want to check derivative orderings for other 2 arg nodes
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeDivide* clone_impl() const override;
      
            NodeDivide* rnd_clone_impl() const override;
    };
}	

#endif
