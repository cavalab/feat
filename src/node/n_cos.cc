/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_cos.h"

namespace FT{
	
    NodeCos::NodeCos(vector<double> W0)
    {
	    name = "cos";
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

    /// Evaluates the node and updates the stack states. 
    void NodeCos::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(cos(W[0] * stack.f.pop())));
    }

    /// Evaluates the node symbolically
    void NodeCos::eval_eqn(Stacks& stack)
    {
        stack.fs.push("cos(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeCos::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return stack.f[stack.f.size()-1] * -sin(W[0] * stack.f[stack.f.size() - 1]);
            case 0: // d/dx0
            default:
               return W[0] * -sin(W[0] * stack.f[stack.f.size() - 1]);
        } 
    }
    
    NodeCos* NodeCos::clone_impl() const { return new NodeCos(*this); }
      
    NodeCos* NodeCos::rnd_clone_impl() const { return new NodeCos(); }
}
