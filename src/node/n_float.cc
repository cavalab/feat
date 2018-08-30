/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    NodeFloat::NodeFloat()
    {
	    name = "f";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeFloat::evaluate(const Data& data, Stacks& stack)
    {
        stack.f.push(stack.b.pop().cast<double>());
    }

    /// Evaluates the node symbolically
    void NodeFloat::eval_eqn(Stacks& stack)
    {
        stack.fs.push("f(" + stack.bs.pop() + ")");
    }
    
    NodeFloat* NodeFloat::clone_impl() const { return new NodeFloat(*this); }

    NodeFloat* NodeFloat::rnd_clone_impl() const { return new NodeFloat(); }  
}
