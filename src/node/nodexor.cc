/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodexor.h"

namespace FT
{    	
    NodeXor::NodeXor()
    {
        name = "xor";
	    otype = 'b';
	    arity['f'] = 0;
	    arity['b'] = 2;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeXor::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXb x1 = stack.b.pop();
        ArrayXb x2 = stack.b.pop();

        ArrayXb res = (x1 != x2).select(ArrayXb::Ones(x1.size()), ArrayXb::Zero(x1.size()));

        stack.b.push(res);
        
    }

    /// Evaluates the node symbolically
    void NodeXor::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.bs.pop() + " XOR " + stack.bs.pop() + ")");
    }
    
    NodeXor* NodeXor::clone_impl() const { return new NodeXor(*this); }

    NodeXor* NodeXor::rnd_clone_impl() const { return new NodeXor(); }  

}
