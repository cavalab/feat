/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SUBTRACT
#define NODE_SUBTRACT

#include "../n_Dx.h"

namespace FT{
	class NodeSubtract : public NodeDx
    {
    	public:
    	
    		NodeSubtract(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeSubtract* clone_impl() const override;
            NodeSubtract* rnd_clone_impl() const override;
    };
}	

#endif
