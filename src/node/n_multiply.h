/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MULTIPLY
#define NODE_MULTIPLY

#include "n_Dx.h"

namespace FT{
	class NodeMultiply : public NodeDx
    {
    	public:
    	
    		NodeMultiply(vector<double> W0 = vector<double>());

            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);
            
            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeMultiply* clone_impl() const override;
            
            NodeMultiply* rnd_clone_impl() const override;
    };
}	

#endif
