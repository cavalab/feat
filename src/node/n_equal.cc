/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_equal.h"
    	  
namespace FT{
 	
    NodeEqual::NodeEqual()
    {
	    name = "=";
	    otype = 'b';
	    arity['f'] = 2;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeEqual::evaluate(Data& data, Stacks& stack)
    {
        stack.push<bool>(stack.pop<double>() == stack.pop<double>());
    }

    /// Evaluates the node symbolically
    void NodeEqual::eval_eqn(Stacks& stack)
    {
        stack.push<bool>("(" + stack.popStr<double>() + "==" + stack.popStr<double>() + ")");
    }
    
    NodeEqual* NodeEqual::clone_impl() const { return new NodeEqual(*this); }

    NodeEqual* NodeEqual::rnd_clone_impl() const { return new NodeEqual(); }  

}
