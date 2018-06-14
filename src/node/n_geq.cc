/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_geq.h"
    	
namespace FT{

    NodeGEQ::NodeGEQ()
    {
	    name = ">=";
	    otype = 'b';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeGEQ::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        stack.b.push(x1 >= x2);
    }

    /// Evaluates the node symbolically
    void NodeGEQ::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + ">=" + stack.fs.pop() + ")");
    }
    
    NodeGEQ* NodeGEQ::clone_impl() const { return new NodeGEQ(*this); }
      
    NodeGEQ* NodeGEQ::rnd_clone_impl() const { return new NodeGEQ(); } 
}
