/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_square.h"
    	
namespace FT{

    NodeSquare::NodeSquare(vector<double> W0)
    {
	    name = "^2";
	    otype = 'f';
	    arity['f'] = 1;
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
    /// Evaluates the node and updates the stack states. 
    void NodeSquare::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(pow(W[0]*stack.f.pop(),2)));
    }
#else
    void NodeSquare::evaluate(Data& data, Stacks& stack)
    {
        GPU_Square(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeSquare::eval_eqn(Stacks& stack)
    {
        stack.fs.push("(" + stack.fs.pop() + "^2)");
    }

    ArrayXd NodeSquare::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return 2 * pow(stack.f[stack.f.size()-1], 2) * this->W[0];
            case 0: // d/dx0
            default:
               return 2 * pow(this->W[0], 2) * stack.f[stack.f.size()-1];
        } 
    }

    NodeSquare* NodeSquare::clone_impl() const { return new NodeSquare(*this); }

    NodeSquare* NodeSquare::rnd_clone_impl() const { return new NodeSquare(); }  
}
