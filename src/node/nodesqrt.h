/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQRT
#define NODE_SQRT

#include "nodeDx.h"

namespace FT{
	class NodeSqrt : public NodeDx
    {
    	public:
    	
    		NodeSqrt(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            ArrayXd getDerivative(Trace& stack, int loc);

        protected:
            NodeSqrt* clone_impl() const override;  
            NodeSqrt* rnd_clone_impl() const override;  
    };
}	

#endif
