/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "../n_Dx.h"

namespace FT{
    
    namespace Pop{
        namespace Op{
	        class NodeIfThenElse : public NodeDx
            {
            	public:
            	
            		NodeIfThenElse();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);            
                    
                    ArrayXf getDerivative(Trace& state, int loc); 

                protected:
                    NodeIfThenElse* clone_impl() const override;
                    NodeIfThenElse* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
