/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_xor.h"

namespace FT
{    	
    NodeXor::NodeXor()
    {
        name = "xor";
	    otype = 'b';
	    arity['b'] = 2;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeXor::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXb x1 = stack.pop<bool>();
        ArrayXb x2 = stack.pop<bool>();

        ArrayXb res = (x1 != x2).select(ArrayXb::Ones(x1.size()), ArrayXb::Zero(x1.size()));

        stack.push<bool>(res);
        
    }

    /// Evaluates the node symbolically
    void NodeXor::eval_eqn(Stacks& stack)
    {
        stack.push<bool>("(" + stack.popStr<bool>() + " XOR " + stack.popStr<bool>() + ")");
    }
    
    NodeXor* NodeXor::clone_impl() const { return new NodeXor(*this); }

    NodeXor* NodeXor::rnd_clone_impl() const { return new NodeXor(); }  

}
