/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IF
#define NODE_IF

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeIf : public NodeDx
            {
            	public:
            	   	
            		NodeIf(vector<float> W0 = vector<float>());
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                    ArrayXf getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeIf* clone_impl() const override;
              
                    NodeIf* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
