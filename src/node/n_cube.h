/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "n_Dx.h"

namespace FT{
	class NodeCube : public NodeDx
    {
    	public:
    		  
    		NodeCube(vector<double> W0 = vector<double>());
    		
            /// Evaluates the node and updates the stack states.  
            void evaluate(Data& data, Stacks& stack);
            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
            ArrayXd getDerivative(Trace& stack, int loc);
            
        protected:
            NodeCube* clone_impl() const override;
            
            NodeCube* rnd_clone_impl() const override;
    };
}	

#endif

