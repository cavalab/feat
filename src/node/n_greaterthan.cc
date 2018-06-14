/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_greaterthan.h"
    	   
namespace FT{
	
    NodeGreaterThan::NodeGreaterThan()
    {
	    name = ">";
	    otype = 'b';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeGreaterThan::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        stack.b.push(x1 > x2);
    }

    /// Evaluates the node symbolically
    void NodeGreaterThan::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + ">" + stack.fs.pop() + ")");
    }
    
    NodeGreaterThan* NodeGreaterThan::clone_impl() const { return new NodeGreaterThan(*this); }
      
    NodeGreaterThan* NodeGreaterThan::rnd_clone_impl() const { return new NodeGreaterThan(); } 
}
