/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_kurtosis.h"
#include "../../utils.h"

namespace FT{

    NodeKurtosis::NodeKurtosis()
    {
        name = "kurtosis";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 1;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeKurtosis::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = kurtosis(limited(stack.z.top().first[x]));
            
        stack.z.pop();
        stack.push<double>(tmp);
        
    }
#else
    void NodeKurtosis::evaluate(const Data& data, Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = kurtosis(stack.z.top().first[x]);
            
        stack.z.pop();
        
    }
#endif

    /// Evaluates the node symbolically
    void NodeKurtosis::eval_eqn(Stacks& stack)
    {
        stack.push<double>("kurtosis(" + stack.zs.pop() + ")");
    }
    
    NodeKurtosis* NodeKurtosis::clone_impl() const { return new NodeKurtosis(*this); }

    NodeKurtosis* NodeKurtosis::rnd_clone_impl() const { return new NodeKurtosis(); } 
}
