/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_RELU
#define NODE_RELU

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace NodeSpace{
        	class NodeRelu : public NodeDx
            {
            	public:
            	  	
            		NodeRelu(vector<double> W0 = vector<double>());
            		
                    /// Evaluates the node and updates the stack states. 
                     void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                     void eval_eqn(Stacks& stack);
                     
                     ArrayXd getDerivative(Trace& stack, int loc);

                protected:
                    NodeRelu* clone_impl() const override;

                    NodeRelu* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
