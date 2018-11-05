/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_2DGAUSSIAN
#define NODE_2DGAUSSIAN

#include <numeric>

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class Node2dGaussian : public NodeDx
            {
            	public:
            	
            		Node2dGaussian(vector<double> W0 = vector<double>());
            		    		
                    /// Evaluates the node and updates the stack states. 
                    void evaluate(const Data& data, Stacks& stack);

                    /// Evaluates the node symbolically
                    void eval_eqn(Stacks& stack);

                    ArrayXd getDerivative(Trace& stack, int loc); 

                protected:
                        Node2dGaussian* clone_impl() const override;

                        Node2dGaussian* rnd_clone_impl() const override;

            };
        }
    }
}	

#endif
