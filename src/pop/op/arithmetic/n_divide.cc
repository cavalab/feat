/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_divide.h"
    	  	
namespace FT{

    namespace Pop{
        namespace Op{
            NodeDivide::NodeDivide(vector<float> W0)
            {
	            name = "/";
	            otype = 'f';
	            arity['f'] = 2;
	            arity['b'] = 0;
	            complexity = 2;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            void NodeDivide::evaluate(const Data& data, State& state)
            {
                ArrayXf x1 = state.pop<float>();
                ArrayXf x2 = state.pop<float>();
                // safe division returns x1/x2 if x2 != 0, and MAX_FLT otherwise               
                ArrayXf ret = (this->W[0] * x1) / (this->W[1] * x2);
                clean(ret);
                state.push<float>(ret); 
            }
            #else
            void NodeDivide::evaluate(const Data& data, State& state)
            {
                GPU_Divide(state.dev_f, state.idx[otype], state.N, W[0], W[1]);
            }
            #endif

            /// Evaluates the node symbolically
            void NodeDivide::eval_eqn(State& state)
            {
                state.push<float>("(" + to_string(W[0], 4) + "*" + state.popStr<float>() + "/" 
                                  + to_string(W[1], 4) + "*" + state.popStr<float>() + ")");            	
            }

            // Might want to check derivative orderings for other 2 arg nodes
            ArrayXf NodeDivide::getDerivative(Trace& state, int loc)
            {
                ArrayXf& x1 = state.get<float>()[state.size<float>()-1];
                ArrayXf& x2 = state.get<float>()[state.size<float>()-2];
                
                switch (loc) {
                    case 3: // d/dW[1]
                        return limited(-this->W[0] * x1/(x2 * pow(this->W[1], 2)));
                    case 2: // d/dW[0]
                        return limited(x1/(this->W[1] * x2));
                    case 1: // d/dx2 
                    {
                        /* std::cout << "x1: " << x1.transpose() << "\n"; */
                        /* ArrayXf num = -this->W[0] * x1; */
                        /* ArrayXf denom = limited(this->W[1] * pow(x2, 2)); */
                        /* ArrayXf val = num/denom; */
                        return limited((-this->W[0] * x1)/(this->W[1] * pow(x2, 2)));
                    }
                    case 0: // d/dx1 
                    default:
                        return limited(this->W[0]/(this->W[1] * x2));
                       // return limited(this->W[1]/(this->W[0] * x2));
                } 
            }
            
            NodeDivide* NodeDivide::clone_impl() const { return new NodeDivide(*this); }
              
            NodeDivide* NodeDivide::rnd_clone_impl() const { return new NodeDivide(); } 
        }
    }
}
