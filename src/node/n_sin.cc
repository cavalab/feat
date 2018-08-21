/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sin.h"
    	
namespace FT{

    NodeSin::NodeSin(vector<double> W0)
    {
	    name = "sin";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 0;
	    complexity = 3;

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
    void NodeSin::evaluate(Data& data, Stacks& stack)
    {

        stack.f.push(limited(sin(W[0]*stack.f.pop())));
    }
#else
    void NodeSin::evaluate(Data& data, Stacks& stack)
    {
        GPU_Sin(stack.dev_f, stack.idx[otype], stack.N, W[0]);
    }
#endif

    /// Evaluates the node symbolically
    void NodeSin::eval_eqn(Stacks& stack)
    {
        stack.fs.push("sin(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeSin::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return stack.f[stack.f.size() - 1] * cos(W[0] * stack.f[stack.f.size()-1]);
            case 0: // d/dx0
            default:
                return W[0] * cos(W[0] * stack.f[stack.f.size() - 1]);
        } 
    }

    // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
    //     switch (loc) {
    //         case 0:
    //         default:
    //             gradients.push_back(W[0] * cos(W[0] * stack_f[stack_f.size() - 2]));
    //     } 
    // }

    // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
    //     update_value = 1
    //     for(auto g : gradients) {
    //         update_value *= g;
    //     }
         
    //     d_w = stack_f[stack_f.size() - 1] * cos(W[0] * stack_f[stack_f.size()-1]);
    //     W[0] = W[0] - n/update_value.size * sum(d_w * update_value);
    // }
    NodeSin* NodeSin::clone_impl() const { return new NodeSin(*this); }

    NodeSin* NodeSin::rnd_clone_impl() const { return new NodeSin(); }  
}
