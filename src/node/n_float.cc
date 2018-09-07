/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    bool isCategorical;

    NodeFloat::NodeFloat(bool isCategorical)
    {
        this->isCategorical = isCategorical;

        if(isCategorical)
        {
	        name = "c2f";
            arity['b'] = 0;
            arity['c'] = 1;
	    }
	    else
	    {
	        name = "b2f";
            arity['b'] = 1;
            arity['c'] = 0;
	    }

	    otype = 'f';
        complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeFloat::evaluate(const Data& data, Stacks& stack)
    {
        if(isCategorical)
            stack.push<double>(stack.pop<int>().cast<double>());
        else
            stack.push<double>(stack.pop<bool>().cast<double>());
    }

    /// Evaluates the node symbolically
    void NodeFloat::eval_eqn(Stacks& stack)
    {
        if(isCategorical)
            stack.push<double>("float(" + stack.popStr<int>() + ")");
        else
            stack.push<double>("float(" + stack.popStr<bool>() + ")");
    }
    
    NodeFloat* NodeFloat::clone_impl() const { return new NodeFloat(*this); }

    NodeFloat* NodeFloat::rnd_clone_impl() const { return new NodeFloat(); }  
}
