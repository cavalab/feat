/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    NodeFloat::NodeFloat(bool isCategorical)
    {
        if(!isCategorical)
        {
	        name = "f";
	        otype = 'f';
	    }
	    else
	    {
	        name = "f_c";
            otype = 'c';
	    }
	    
        arity['b'] = 1;
        complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeFloat::evaluate(Data& data, Stacks& stack)
    {
        if(otype == 'f')
            stack.push<double>(stack.pop<bool>().cast<double>());
        else
            stack.c.push(stack.pop<bool>().cast<int>());
    }

    /// Evaluates the node symbolically
    void NodeFloat::eval_eqn(Stacks& stack)
    {
        stack.push<double>("f(" + stack.popStr<bool>() + ")");
    }
    
    NodeFloat* NodeFloat::clone_impl() const { return new NodeFloat(*this); }

    NodeFloat* NodeFloat::rnd_clone_impl() const { return new NodeFloat(); }  
}
