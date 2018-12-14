/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeDivide : public NodeDx
            {
            	public:
            	  	
            		NodeDivide(vector<float> W0 = vector<float>());
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    // Might want to check derivative orderings for other 2 arg nodes
                    ArrayXf getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeDivide* clone_impl() const override;
              
                    NodeDivide* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
