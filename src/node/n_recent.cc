/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_recent.h"

namespace FT{

    NodeRecent::NodeRecent()
    {
        name = "recent";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeRecent::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
        {
            // find max time
            ArrayXd::Index maxIdx; 
            double maxtime = stack.z.top().second[x].maxCoeff(&maxIdx);
            // return value at max time 
            tmp(x) = stack.z.top().first[x](maxIdx);
        }

        stack.z.pop();
        
        stack.push<double>(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeRecent::eval_eqn(Stacks& stack)
    {
        stack.push<double>("recent(" + stack.zs.pop() + ")");
    }
    
    NodeRecent* NodeRecent::clone_impl() const { return new NodeRecent(*this); }

    NodeRecent* NodeRecent::rnd_clone_impl() const { return new NodeRecent(); }
}
