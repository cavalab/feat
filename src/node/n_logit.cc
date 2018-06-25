/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_logit.h"
    	
namespace FT{

    NodeLogit::NodeLogit(vector<double> W0)
    {
        name = "logit";
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
    void NodeLogit::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(1/(1+(limited(exp(-W[0]*stack.f.pop())))));
    }
#else
    void NodeLogit::evaluate(Data& data, Stacks& stack)
    {
        GPU_Logit(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeLogit::eval_eqn(Stacks& stack)
    {
        /* stack.fs.push("1/(1+exp(-" + stack.fs.pop() + "))"); */
        stack.fs.push("logit(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeLogit::getDerivative(Trace& stack, int loc) {
        ArrayXd numerator, denom;
        switch (loc) {
            case 1: // d/dw0
                numerator = stack.f[stack.f.size() -1] * exp(-W[0] * stack.f[stack.f.size() -1]);
                denom = pow(1 + exp(-W[0] * stack.f[stack.f.size()-1]), 2);
                return numerator/denom;
            case 0: // d/dx0
            default:
                numerator = W[0] * exp(-W[0] * stack.f[stack.f.size() - 1]);
                denom = pow(1 + exp(-W[0] * stack.f[stack.f.size() - 1]), 2);
                return numerator/denom;
        } 
    }
    
    NodeLogit* NodeLogit::clone_impl() const { return new NodeLogit(*this); }

    NodeLogit* NodeLogit::rnd_clone_impl() const { return new NodeLogit(); }
}
