/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_multiply.h"
    	
namespace FT{

    NodeMultiply::NodeMultiply(vector<double> W0)
    {
	    name = "*";
	    otype = 'f';
	    arity['f'] = 2;
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

    /// Evaluates the node and updates the stack states. 
    void NodeMultiply::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(limited(W[0]*stack.pop<double>() * W[1]*stack.pop<double>()));
    }

    /// Evaluates the node symbolically
    void NodeMultiply::eval_eqn(Stacks& stack)
    {
	    stack.push<double>("(" + stack.popStr<double>() + "*" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeMultiply::getDerivative(Trace& stack, int loc)
    {
        ArrayXd& x1 = stack.get<double>()[stack.size<double>()-1];
        ArrayXd& x2 = stack.get<double>()[stack.size<double>()-2];
        
        switch (loc) {
            case 3: // d/dW[1]
                return x1 * this->W[0] * x2;
            case 2: // d/dW[0] 
                return x1 * this->W[1] * x2;
            case 1: // d/dx2
                return this->W[0] * this->W[1] * x2;
            case 0: // d/dx1
            default:
                return this->W[1] * this->W[0] * x1;
        } 
    }
    
    NodeMultiply* NodeMultiply::clone_impl() const { return new NodeMultiply(*this); }

    NodeMultiply* NodeMultiply::rnd_clone_impl() const { return new NodeMultiply(); }  

}
