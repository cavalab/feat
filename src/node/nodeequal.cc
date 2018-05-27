/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "nodeequal.h"
    	  
namespace FT{
 	
    NodeEqual::NodeEqual()
    {
	    name = "=";
	    otype = 'b';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeEqual::evaluate(Data& data, Stacks& stack)
    {
        stack.b.push(stack.f.pop() == stack.f.pop());
    }

    /// Evaluates the node symbolically
    void NodeEqual::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + "==" + stack.fs.pop() + ")");
    }
    
    NodeEqual* NodeEqual::clone_impl() const { return new NodeEqual(*this); }

    NodeEqual* NodeEqual::rnd_clone_impl() const { return new NodeEqual(); }  

}
