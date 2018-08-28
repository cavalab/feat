/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_if.h"
    	
namespace FT{
       	
    NodeIf::NodeIf(vector<double> W0)
    {
		name = "if";
		otype = 'f';
		arity['f'] = 1;
		arity['b'] = 1;
		complexity = 5;
        W.push_back(0);
	}

    /// Evaluates the node and updates the stack states. 
    void NodeIf::evaluate(Data& data, Stacks& stack)
    {
        stack.push<double>(limited(stack.pop<bool>().select(stack.pop<double>(),0)));
    }

    /// Evaluates the node symbolically
    void NodeIf::eval_eqn(Stacks& stack)
    {
      stack.push<double>("if(" + stack.popStr<bool>() + "," + stack.popStr<double>() + "," + "0)");
    }
    
    ArrayXd NodeIf::getDerivative(Trace& stack, int loc) 
    {
        ArrayXd& xf = stack.get<double>()[stack.size<double>()-1];
        ArrayXb& xb = stack.get<bool>()[stack.size<bool>()-1];
        
        switch (loc) {
            case 1: // d/dW[0]
                return ArrayXd::Zero(xf.size()); 
            case 0: // d/dx1
            default:
                return xb.cast<double>(); 
                /* .select(ArrayXd::Ones(stack.f[stack.f.size()-1].size(), */
                /*                  ArrayXd::Zero(stack.f[stack.f.size()-1].size()); */
        } 
    }
    
    NodeIf* NodeIf::clone_impl() const { return new NodeIf(*this); }
      
    NodeIf* NodeIf::rnd_clone_impl() const { return new NodeIf(); }
}
