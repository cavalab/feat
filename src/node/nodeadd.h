/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_ADD
#define NODE_ADD

#include "nodeDx.h"

namespace FT{
	class NodeAdd : public NodeDx
    {
    	public:
    	
    		NodeAdd(vector<double> W0 = vector<double>());
    		    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);
			
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            // NEED TO MAKE SURE CASE 0 IS TOP OF STACK, CASE 2 IS w[0]
            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc);
            
        protected:
            NodeAdd* clone_impl() const override;
      
            NodeAdd* rnd_clone_impl() const override;
    };
}	

#endif
