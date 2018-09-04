/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_gaussian.h"
    	
namespace FT{

    NodeGaussian::NodeGaussian(vector<double> W0)
    {
        name = "gaussian";
		otype = 'f';
		arity['f'] = 1;
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
    void NodeGaussian::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(limited(exp(-pow(W[0] - stack.pop<double>(), 2))));
    }

    /// Evaluates the node symbolically
    void NodeGaussian::eval_eqn(Stacks& stack)
    {
        stack.push<double>("gauss(" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeGaussian::getDerivative(Trace& stack, int loc) 
    {
        ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
        
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
