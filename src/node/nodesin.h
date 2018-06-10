/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIN
#define NODE_SIN

#include "nodeDx.h"

namespace FT{
	class NodeSin : public NodeDx
    {
    	public:
    	
    		NodeSin(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeSin* clone_impl() const override;  
            NodeSin* rnd_clone_impl() const override;  
    };
}	

#endif
