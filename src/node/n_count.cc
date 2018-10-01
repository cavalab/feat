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
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeCount::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).cols();
          
        stack.z.pop();
        
        stack.push<double>(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeCount::eval_eqn(Stacks& stack)
    {
        stack.push<double>("count(" + stack.zs.pop() + ")");
    }
    
    NodeCount*NodeCount::clone_impl() const { return new NodeCount(*this); }
     
    NodeCount* NodeCount::rnd_clone_impl() const { return new NodeCount(); }
}
