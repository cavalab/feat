/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_exponential.h"
   	
namespace FT{

    NodeExponential::NodeExponential(vector<double> W0)
    {
	    name = "exp";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 0;
	    complexity = 4;

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
    /// Evaluates the node and updates the stack states. 
    void NodeExponential::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(limited(exp(this->W[0] * stack.pop<double>())));
    }
#else
    void NodeExponential::evaluate(const Data& data, Stacks& stack)
    {
        GPU_Exp(stack.dev_f, stack.idx[otype], stack.N, W[0]);
    }
#endif

    /// Evaluates the node symbolically
    void NodeExponential::eval_eqn(Stacks& stack)
    {
        stack.push<double>("exp(" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeExponential::getDerivative(Trace& stack, int loc)
    {
        ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
        
        switch (loc) {
            case 1: // d/dw0
                return x * limited(exp(this->W[0] * x));
            case 0: // d/dx0
            default:
               return this->W[0] * limited(exp(W[0] * x));
        } 
    }
    
    NodeExponential* NodeExponential::clone_impl() const { return new NodeExponential(*this); }
      
    NodeExponential* NodeExponential::rnd_clone_impl() const { return new NodeExponential(); }  
}
