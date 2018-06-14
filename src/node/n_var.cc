/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_var.h"
#include "../utils.h"
    	
namespace FT{

    NodeVar::NodeVar()
    {
        name = "variance";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeVar::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        ArrayXd tmp1;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = variance(limited(stack.z.top().first[x]));
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeVar::eval_eqn(Stacks& stack)
    {
        stack.fs.push("variance(" + stack.zs.pop() + ")");
    }

    NodeVar* NodeVar::clone_impl() const { return new NodeVar(*this); }

    NodeVar* NodeVar::rnd_clone_impl() const { return new NodeVar(); } 

}
