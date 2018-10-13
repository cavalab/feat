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
    void NodeSin::evaluate(const Data& data, Stacks& stack)
    {

        stack.push<double>(limited(sin(W[0]*stack.pop<double>())));
    }

    /// Evaluates the node symbolically
    void NodeSin::eval_eqn(Stacks& stack)
    {
        stack.push<double>("sin(" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeSin::getDerivative(Trace& stack, int loc)
    {
        ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
        
        switch (loc) {
            case 1: // d/dw0
                return x * cos(W[0] * x);
            case 0: // d/dx0
            default:
                return W[0] * cos(W[0] * x);
        } 
    }

    NodeSin* NodeSin::clone_impl() const { return new NodeSin(*this); }

    NodeSin* NodeSin::rnd_clone_impl() const { return new NodeSin(); }  
}
