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

    /// Evaluates the node and updates the stack states. 
    void NodeExponential::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(exp(this->W[0] * stack.f.pop())));
    }

    /// Evaluates the node symbolically
    void NodeExponential::eval_eqn(Stacks& stack)
    {
        stack.fs.push("exp(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeExponential::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return stack.f[stack.f.size()-1] * limited(exp(this->W[0] * stack.f[stack.f.size()-1]));
            case 0: // d/dx0
            default:
               return this->W[0] * limited(exp(W[0] * stack.f[stack.f.size()-1]));
        } 
    }
    
    NodeExponential* NodeExponential::clone_impl() const { return new NodeExponential(*this); }
      
    NodeExponential* NodeExponential::rnd_clone_impl() const { return new NodeExponential(); }  
}
