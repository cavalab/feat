/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodemode.h"

namespace FT{
    	
    NodeMode::NodeMode()
    {
        name = "mode";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeMode::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).mean();
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeMode::eval_eqn(Stacks& stack)
    {
        stack.fs.push("mean(" + stack.zs.pop() + ")");
    }
    
    NodeMode* NodeMode::clone_impl() const { return new NodeMode(*this); }

    NodeMode* NodeMode::rnd_clone_impl() const { return new NodeMode(); }  
}
