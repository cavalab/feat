/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOGIT
#define NODE_LOGIT

#include "nodeDx.h"

namespace FT{
	class NodeLogit : public NodeDx
    {
    	public:
    	
    		NodeLogit(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc);

        protected:
            NodeLogit* clone_impl() const override;

            NodeLogit* rnd_clone_impl() const override;
    };
}	

#endif
