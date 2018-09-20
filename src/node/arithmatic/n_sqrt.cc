/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sqrt.h"

namespace FT{
    		
    NodeSqrt::NodeSqrt(vector<double> W0)
    {
        name = "sqrt";
	    otype = 'f';
	    arity['f'] = 1;
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
    /// Evaluates the node and updates the stack states. 
    void NodeSqrt::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(sqrt(W[0]*stack.pop<double>().abs()));
    }
#else
    void NodeSqrt::evaluate(const Data& data, Stacks& stack)
    {
        GPU_Sqrt(stack.dev_f, stack.idx[otype], stack.N, W[0]);
    }
#endif

    /// Evaluates the node symbolically
    void NodeSqrt::eval_eqn(Stacks& stack)
    {
        stack.push<double>("sqrt(|" + stack.popStr<double>() + "|)");
    }

    ArrayXd NodeSqrt::getDerivative(Trace& stack, int loc)
    {
        ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
        
        switch (loc) {
            case 1: // d/dw0
                return limited(x / (2 * sqrt(this->W[0] * x)));
            case 0: // d/dx0
            default:
                return limited(this->W[0] / (2 * sqrt(this->W[0] * x)));
        } 
    }

    NodeSqrt* NodeSqrt::clone_impl() const { return new NodeSqrt(*this); }

    NodeSqrt* NodeSqrt::rnd_clone_impl() const { return new NodeSqrt(); }  
}
