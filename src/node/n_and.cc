/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_and.h"

namespace FT{

    NodeAnd::NodeAnd()
    {
	    name = "and";
	    otype = 'b';
	    arity['f'] = 0;
	    arity['b'] = 2;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeAnd::evaluate(Data& data, Stacks& stack)
    {
        stack.b.push(stack.b.pop() && stack.b.pop());

    }

    /// Evaluates the node symbolically
    void NodeAnd::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.bs.pop() + " AND " + stack.bs.pop() + ")");
    }
    
    NodeAnd* NodeAnd::clone_impl() const { return new NodeAnd(*this); }
      
    NodeAnd* NodeAnd::rnd_clone_impl() const { return new NodeAnd(); } 
}
