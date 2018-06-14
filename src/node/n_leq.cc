/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_leq.h"
    	
namespace FT{

    NodeLEQ::NodeLEQ()
    {
	    name = "<=";
	    otype = 'b';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeLEQ::evaluate(Data& data, Stacks& stack)
    {
      	ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        stack.b.push(x1 <= x2);
    }

    /// Evaluates the node symbolically
    void NodeLEQ::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + "<=" + stack.fs.pop() + ")");
    }
    
    NodeLEQ* NodeLEQ::clone_impl() const { return new NodeLEQ(*this); }  

    NodeLEQ* NodeLEQ::rnd_clone_impl() const { return new NodeLEQ(); }
}
