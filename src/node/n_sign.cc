/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sign.h"

namespace FT{
    NodeSign::NodeSign(vector<double> W0)
    {
        name = "sign";
	    otype = 'f';
	    arity['f'] = 1;
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
    void NodeSign::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXd x = stack.f.pop();
        ArrayXd ones = ArrayXd::Ones(x.size());

	    ArrayXd res = (W[0] * x > 0).select(ones, 
                                            (x == 0).select(ArrayXd::Zero(x.size()), 
                                                            -1*ones)); 
        stack.f.push(res);
    }
#else
    void NodeSign::evaluate(Data& data, Stacks& stack)
    {
        GPU_Sign(stack.dev_f, stack.idx[otype], stack.N, W[0]);
    }
#endif

    /// Evaluates the node symbolically
    void NodeSign::eval_eqn(Stacks& stack)
    {
        stack.fs.push("sign("+ stack.fs.pop() +")");
    }

    ArrayXd NodeSign::getDerivative(Trace& stack, int loc) {
        // Might want to experiment with using a perceptron update rule or estimating with some other function
        switch (loc) {
            case 1: // d/dw0
                return stack.f[stack.f.size()-1] / (2 * sqrt(W[0] * stack.f[stack.f.size()-1]));
            case 0: // d/dx0
            default:
                return W[0] / (2 * sqrt((W[0] * stack.f[stack.f.size()-1]).abs()));
        } 
    }
    
    NodeSign* NodeSign::clone_impl() const { return new NodeSign(*this); }

    NodeSign* NodeSign::rnd_clone_impl() const { return new NodeSign(); }  
}
