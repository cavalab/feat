/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_TANH
#define NODE_TANH

#include "n_Dx.h"

namespace FT{
	class NodeTanh : public NodeDx
    {
    	public:
    	
    		NodeTanh(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeTanh* clone_impl() const override;
            NodeTanh* rnd_clone_impl() const override;
    };
}	

#endif
