/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_count.h"
    	
namespace FT{

    NodeCount::NodeCount()
    {
        name = "count";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

#ifndef USE_CUDA 
    /// Evaluates the node and updates the stack states. 
    void NodeCount::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).cols();
          
        stack.z.pop();
        
        stack.f.push(tmp);
        
    }
#else
    void NodeCount::evaluate(Data& data, Stacks& stack)
    {
        
        for(unsigned x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = limited(stack.z.top().first[x]).cols();
          
        stack.z.pop();
        
    }
#endif

    /// Evaluates the node symbolically
    void NodeCount::eval_eqn(Stacks& stack)
    {
        stack.fs.push("count(" + stack.zs.pop() + ")");
    }
    
    NodeCount*NodeCount::clone_impl() const { return new NodeCount(*this); }
     
    NodeCount* NodeCount::rnd_clone_impl() const { return new NodeCount(); }
}
