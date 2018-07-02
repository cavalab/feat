/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_add.h"

namespace FT{

    NodeAdd::NodeAdd(vector<double> W0)
    {
	    name = "+";
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
    void NodeAdd::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(this->W[0] * stack.f.pop() + this->W[1] * stack.f.pop()));
    }
#else
    void NodeAdd::evaluate(Data& data, Stacks& stack)
	{
        GPU_Add(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeAdd::eval_eqn(Stacks& stack)
    {
        stack.fs.push("(" + stack.fs.pop() + "+" + stack.fs.pop() + ")");
    }

    // NEED TO MAKE SURE CASE 0 IS TOP OF STACK, CASE 2 IS w[0]
    ArrayXd NodeAdd::getDerivative(Trace& stack, int loc) 
    {
        switch (loc) {
            case 3: // d/dW[1] 
                return stack.f[stack.f.size()-2];
            case 2: // d/dW[0]
                return stack.f[stack.f.size()-1];
            case 1: // d/dx2
                return this->W[1] * ArrayXd::Ones(stack.f[stack.f.size()-2].size());
            case 0: // d/dx1
            default:
                return this->W[0] * ArrayXd::Ones(stack.f[stack.f.size()-1].size());
        } 
    }
    
    NodeAdd* NodeAdd::clone_impl() const { return new NodeAdd(*this); }
      
    NodeAdd* NodeAdd::rnd_clone_impl() const { return new NodeAdd(); }
}
