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
            	
            		NodeLog(vector<double> W0 = vector<double>());
            		
                    /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);

                    ArrayXd getDerivative(Trace& state, int loc);
                    
                protected:
                    NodeLog* clone_impl() const override;

                    NodeLog* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
