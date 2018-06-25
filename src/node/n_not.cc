/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_not.h"
    	
namespace FT{

    NodeNot::NodeNot()
    {
	    name = "not";
	    otype = 'b';
	    arity['f'] = 0;
	    arity['b'] = 1;
	    complexity = 1;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeNot::evaluate(Data& data, Stacks& stack)
    {
        stack.b.push(!stack.b.pop());
    }
#else
    void NodeNot::evaluate(Data& data, Stacks& stack)
    {
        GPU_Not(stack.dev_b, stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeNot::eval_eqn(Stacks& stack)
    {
        stack.bs.push("NOT(" + stack.bs.pop() + ")");
    }
    
    NodeNot* NodeNot::clone_impl() const { return new NodeNot(*this); }

    NodeNot* NodeNot::rnd_clone_impl() const { return new NodeNot(); }  
}
