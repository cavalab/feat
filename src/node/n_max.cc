/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_max.h"

namespace FT{

    NodeMax::NodeMax()
    {
        name = "max";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeMax::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).maxCoeff();

        stack.z.pop();
        
        stack.push<double>(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeMax::eval_eqn(Stacks& stack)
    {
        stack.push<double>("max(" + stack.zs.pop() + ")");
    }
    
    NodeMax* NodeMax::clone_impl() const { return new NodeMax(*this); }

    NodeMax* NodeMax::rnd_clone_impl() const { return new NodeMax(); }
}
