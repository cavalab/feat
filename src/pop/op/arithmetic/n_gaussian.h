/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeGaussian : public NodeDx
            {
            	public:

            		NodeGaussian(vector<float> W0 = vector<float>());
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    ArrayXf getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeGaussian* clone_impl() const override;
              
                    NodeGaussian* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
