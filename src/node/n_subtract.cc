/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_subtract.h"
    	    	
namespace FT{

    NodeSubtract::NodeSubtract(vector<double> W0)
    {
	    name = "-";
	    otype = 'f';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 1;

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
    void NodeSubtract::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(W[0]*stack.f.pop() - W[1]*stack.f.pop()));
    }
#else
    void NodeSubtract::evaluate(Data& data, Stacks& stack)
    {
        GPU_Subtract(stack.dev_f, stack.idx[otype], stack.N, W[0], W[1]);
    }
#endif

    /// Evaluates the node symbolically
    void NodeSubtract::eval_eqn(Stacks& stack)
    {
        stack.fs.push("(" + stack.fs.pop() + "-" + stack.fs.pop() + ")");
    }

    ArrayXd NodeSubtract::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 3: // d/dW[1]
                return -stack.f[stack.f.size()-2];
            case 2: // d/dW[0]
                return stack.f[stack.f.size()-1];
            case 1: // d/dx2
                return -this->W[1] * ArrayXd::Ones(stack.f[stack.f.size()-2].size());
            case 0: //d/dx1
            default:
               return this->W[0] * ArrayXd::Ones(stack.f[stack.f.size()-1].size());
        } 
    }

    NodeSubtract* NodeSubtract::clone_impl() const { return new NodeSubtract(*this); }

    NodeSubtract* NodeSubtract::rnd_clone_impl() const { return new NodeSubtract(); }  
}
