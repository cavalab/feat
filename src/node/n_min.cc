/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_min.h"
    	
namespace FT{

    NodeMin::NodeMin()
    {
        name = "min";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeMin::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).minCoeff();
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }
#else
    void NodeMin::evaluate(Data& data, Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = stack.z.top().first[x].minCoeff();
            
        stack.z.pop();

        
    }
#endif

    /// Evaluates the node symbolically
    void NodeMin::eval_eqn(Stacks& stack)
    {
        stack.fs.push("min(" + stack.zs.pop() + ")");
    }
    
    NodeMin* NodeMin::clone_impl() const { return new NodeMin(*this); }

    NodeMin* NodeMin::rnd_clone_impl() const { return new NodeMin(); } 
}
