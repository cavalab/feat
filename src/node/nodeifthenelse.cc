/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodeifthenelse.h"

namespace FT{

    NodeIfThenElse::NodeIfThenElse()
    {
	    name = "ite";
	    otype = 'f';
	    arity['f'] = 2;
	    arity['b'] = 1;
	    complexity = 5;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeIfThenElse::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd f1 = stack.f.pop();
        ArrayXd f2 = stack.f.pop();
        stack.f.push(limited(stack.b.pop().select(f1,f2)));
    }

    /// Evaluates the node symbolically
    void NodeIfThenElse::eval_eqn(Stacks& stack)
    {
        stack.fs.push("if-then-else(" + stack.bs.pop() + "," + stack.fs.pop() + "," + stack.fs.pop() + ")");
    }
    
    NodeIfThenElse* NodeIfThenElse::clone_impl() const { return new NodeIfThenElse(*this); }

    NodeIfThenElse* NodeIfThenElse::rnd_clone_impl() const { return new NodeIfThenElse(); }  
}
