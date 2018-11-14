/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sign.h"

namespace FT{

    namespace Pop{
        namespace Op{
            NodeSign::NodeSign()
            {
                name = "sign";
	            otype = 'f';
	            arity['f'] = 1;
	            complexity = 1;

            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeSign::evaluate(const Data& data, State& state)
            {
	            ArrayXd x = state.pop<double>();
                ArrayXd ones = ArrayXd::Ones(x.size());

	            ArrayXd res = ( x > 0).select(ones, 
                                                    (x == 0).select(ArrayXd::Zero(x.size()), 
                                                                    -1*ones)); 
                state.push<double>(res);
            }
            #else
            void NodeSign::evaluate(const Data& data, State& state)
            {
                GPU_Sign(state.dev_f, state.idx[otype], state.N, W[0]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeSign::eval_eqn(State& state)
            {
                state.push<double>("sign("+ state.popStr<double>() +")");
            }

            
            NodeSign* NodeSign::clone_impl() const { return new NodeSign(*this); }

            NodeSign* NodeSign::rnd_clone_impl() const { return new NodeSign(); }  
        }
    }
}
