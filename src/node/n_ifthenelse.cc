/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_ifthenelse.h"

namespace FT{

    NodeIfThenElse::NodeIfThenElse()
    {
		name = "ite";
		otype = 'f';
		arity['f'] = 2;
		arity['b'] = 1;
		complexity = 5;
        W = {0.0, 0.0};
	}

    /// Evaluates the node and updates the stack states. 
    void NodeIfThenElse::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd f1 = stack.pop<double>();
        ArrayXd f2 = stack.pop<double>();
        stack.push<double>(limited(stack.pop<bool>().select(f1,f2)));
    }

    /// Evaluates the node symbolically
    void NodeIfThenElse::eval_eqn(Stacks& stack)
    {
        stack.push<double>("if-then-else(" + stack.popStr<bool>() + 
                           "," + stack.popStr<double>() + "," + 
                           stack.popStr<double>() + ")");
    }
    
    ArrayXd NodeIfThenElse::getDerivative(Trace& stack, int loc) 
    {
        ArrayXd& xf = stack.get<double>()[stack.size<double>()-1];
        ArrayXb& xb = stack.get<bool>()[stack.size<bool>()-1];
        
        switch (loc) {
            case 3: // d/dW[0]
            case 2: 
                return ArrayXd::Zero(xf.size()); 
            case 1: // d/dx2
                return (!xb).cast<double>(); 
            case 0: // d/dx1
            default:
                return xb.cast<double>(); 
                /* .select(ArrayXd::Ones(stack.f[stack.f.size()-1].size(), */
                /*                  ArrayXd::Zero(stack.f[stack.f.size()-1].size()); */
        } 
    }

    
    NodeIfThenElse* NodeIfThenElse::clone_impl() const { return new NodeIfThenElse(*this); }

    NodeIfThenElse* NodeIfThenElse::rnd_clone_impl() const { return new NodeIfThenElse(); }
}
