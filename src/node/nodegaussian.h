/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "nodeDx.h"

namespace FT{
	class NodeGaussian : public NodeDx
    {
    	public:

    		NodeGaussian(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeGaussian* clone_impl() const override;
      
            NodeGaussian* rnd_clone_impl() const override;
    };
}	

#endif
