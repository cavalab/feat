/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodeif.h"
    	
namespace FT{
       	
    NodeIf::NodeIf()
    {
	    name = "if";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 1;
	    complexity = 5;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeIf::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(stack.b.pop().select(stack.f.pop(),0)));
    }

    /// Evaluates the node symbolically
    void NodeIf::eval_eqn(Stacks& stack)
    {
      stack.fs.push("if(" + stack.bs.pop() + "," + stack.fs.pop() + "," + "0)");
    }
    
    NodeIf* NodeIf::clone_impl() const { return new NodeIf(*this); }
      
    NodeIf* NodeIf::rnd_clone_impl() const { return new NodeIf(); }
}
