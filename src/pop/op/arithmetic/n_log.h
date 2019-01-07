/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOG
#define NODE_LOG

#include "../n_Dx.h"

namespace FT{

    namespace Pop{
        namespace Op{
        	class NodeLog : public NodeDx
            {
            	public:
            	
            		NodeLog(vector<float> W0 = vector<float>());
            		
                    /// Safe log: pushes log(abs(x)) or MIN_FLT if x is near zero. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    ArrayXf getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeLog* clone_impl() const override;

                    NodeLog* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
