/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_skew.h"
#include "../utils.h"

namespace FT{
	
    NodeSkew::NodeSkew()
    {
        name = "skew";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 3;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeSkew::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = skew(limited(stack.z.top().first[x]));
            
        stack.z.pop();

        stack.push<double>(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeSkew::eval_eqn(Stacks& stack)
    {
        stack.push<double>("skew(" + stack.zs.pop() + ")");
    }

    NodeSkew* NodeSkew::clone_impl() const { return new NodeSkew(*this); }

    NodeSkew* NodeSkew::rnd_clone_impl() const { return new NodeSkew(); } 
}
