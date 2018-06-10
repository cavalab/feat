/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodegaussian.h"
    	
namespace FT{

    NodeGaussian::NodeGaussian(vector<double> W0)
    {
        name = "gaussian";
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
    void NodeGaussian::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(exp(-pow(W[0] - stack.f.pop(), 2))));
    }

    /// Evaluates the node symbolically
    void NodeGaussian::eval_eqn(Stacks& stack)
    {
        /* stack.fs.push("exp(-(" +std::to_string(W[0]) + '-' + stack.fs.pop() + ")^2)"); */
        stack.fs.push("gauss(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeGaussian::getDerivative(Trace& stack, int loc) 
    {
        ArrayXd& x = stack.f[stack.f.size()-1];
        
        switch (loc) {
            case 1: // d/dw0
                return limited(-2 * (W[0] - 2) * exp(-pow(W[0] - x, 2)));
            case 0: // d/dx0
            default:
                return limited(2 * (W[0] - 2) * exp(-pow(W[0] - x, 2)));
        } 
    }
    
    NodeGaussian* NodeGaussian::clone_impl() const { return new NodeGaussian(*this); }
      
    NodeGaussian* NodeGaussian::rnd_clone_impl() const { return new NodeGaussian(); }  
}
