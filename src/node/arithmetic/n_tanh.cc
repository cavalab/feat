/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_tanh.h"
    	
namespace FT{

    NodeTanh::NodeTanh(vector<double> W0)
    {
        name = "tanh";
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
    void NodeTanh::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(limited(tanh(W[0]*stack.pop<double>())));
    }

    /// Evaluates the node symbolically
    void NodeTanh::eval_eqn(Stacks& stack)
    {
        stack.push<double>("tanh(" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeTanh::getDerivative(Trace& stack, int loc)
    {
        ArrayXd numerator;
        ArrayXd denom;
        ArrayXd x = stack.get<double>()[stack.size<double>()-1];
        switch (loc) {
            case 1: // d/dw0
                numerator = 4 * x * exp(2 * this->W[0] * x);
                denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                // numerator = 4 * x * exp(2 * this->W[0] * x - 1]); 
                // denom = pow(exp(2 * this->W[0] * x) + 1,2);
                return numerator/denom;
            case 0: // d/dx0
            default:
                numerator = 4 * this->W[0] * exp(2 * this->W[0] * x);
                denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                // numerator = 4 * W[0] * exp(2 * W[0] * stack.f[stack.f.size() - 1]);
                // denom = pow(exp(2 * W[0] * stack.f[stack.f.size()-1]),2);
                return numerator/denom;
        } 
    }

    NodeTanh* NodeTanh::clone_impl() const { return new NodeTanh(*this); }
     
    NodeTanh* NodeTanh::rnd_clone_impl() const { return new NodeTanh(); }
}
