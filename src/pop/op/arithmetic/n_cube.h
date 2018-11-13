/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeCube : public NodeDx
            {
            	public:
            		  
            		NodeCube(vector<double> W0 = vector<double>());

                    /// Evaluates the node and updates the state states.  
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    ArrayXd getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeCube* clone_impl() const override;
              
                    NodeCube* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif

