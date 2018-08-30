/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_or.h"
    	
namespace FT{

    NodeOr::NodeOr()
    {
        name = "or";
	    otype = 'b';
	    arity['f'] = 0;
	    arity['b'] = 2;
	    complexity = 2;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeOr::evaluate(const Data& data, Stacks& stack)
    {
        stack.b.push(stack.b.pop() || stack.b.pop());

    }

    /// Evaluates the node symbolically
    void NodeOr::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.bs.pop() + " OR " + stack.bs.pop() + ")");
    }
    
    NodeOr* NodeOr::clone_impl() const { return new NodeOr(*this); }

    NodeOr* NodeOr::rnd_clone_impl() const { return new NodeOr(); }  

}
