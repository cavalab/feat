/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "n_Dx.h"

namespace FT{
	class NodeIfThenElse : public NodeDx
    {
    	public:
    	
    		NodeIfThenElse();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);
            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeIfThenElse* clone_impl() const override;
            NodeIfThenElse* rnd_clone_impl() const override;
    };
}	

#endif
