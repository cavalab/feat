/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_ADD
#define NODE_ADD

#include "n_Dx.h"

namespace FT{
	class NodeAdd : public NodeDx
    {
    	public:
    	
    		NodeAdd(vector<double> W0 = vector<double>());
    		    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);
			
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            ArrayXd getDerivative(Trace& stack, int loc);

        protected:
            NodeAdd* clone_impl() const override;
      
            NodeAdd* rnd_clone_impl() const override;
    };
}	

#endif
