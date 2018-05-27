/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "nodemedian.h"
#include "../utils.h"

namespace FT{

    NodeMedian::NodeMedian()
    {
        name = "median";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeMedian::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = median(limited(stack.z.top().first[x]));
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeMedian::eval_eqn(Stacks& stack)
    {
        stack.fs.push("median(" + stack.zs.pop() + ")");
    }
    
    NodeMedian* NodeMedian::clone_impl() const { return new NodeMedian(*this); }

    NodeMedian* NodeMedian::rnd_clone_impl() const { return new NodeMedian(); } 
}
