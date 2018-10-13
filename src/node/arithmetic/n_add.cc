/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_add.h"

namespace FT{

    NodeAdd::NodeAdd(vector<double> W0)
    {
	    name = "+";
	    otype = 'f';
	    arity['f'] = 2;
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

    /// Evaluates the node and updates the stack states. 
    void NodeAdd::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd x1 = stack.pop<double>();
        ArrayXd x2 = stack.pop<double>();
        stack.push<double>(limited(this->W[0]*x1+this->W[1]*x2));
        /* stack.push<double>(limited(this->W[0]*stack.pop<double>()+this->W[1]*stack.pop<double>())); */
    }

    /// Evaluates the node symbolically
    void NodeAdd::eval_eqn(Stacks& stack)
    {
        stack.push<double>("(" + stack.popStr<double>() + "+" + stack.popStr<double>() + ")");
    }

    // NEED TO MAKE SURE CASE 0 IS TOP OF STACK, CASE 2 IS w[0]
    ArrayXd NodeAdd::getDerivative(Trace& stack, int loc) 
    {
        ArrayXd x1 = stack.get<double>()[stack.size<double>()-1];
        ArrayXd x2 = stack.get<double>()[stack.size<double>()-2];
        
        switch (loc) {
            case 3: // d/dW[1] 
                return x2;
            case 2: // d/dW[0]
                return x1;
            case 1: // d/dx2
                return this->W[1] * ArrayXd::Ones(x2.size());
            case 0: // d/dx1
            default:
                return this->W[0] * ArrayXd::Ones(x1.size());
        } 
    }
    
    NodeAdd* NodeAdd::clone_impl() const { return new NodeAdd(*this); }
      
    NodeAdd* NodeAdd::rnd_clone_impl() const { return new NodeAdd(); }
}
