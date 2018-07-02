/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_lessthan.h"

namespace FT{

    NodeLessThan::NodeLessThan()
    {
	    name = "<";
	    otype = 'b';
	    arity['f'] = 2;
	    arity['b'] = 0;
	    complexity = 2;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeLessThan::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        stack.b.push(x1 < x2);
    }
#else
    void NodeLessThan::evaluate(Data& data, Stacks& stack)
    {
        GPU_LessThan(stack.dev_f, stack.dev_b, stack.idx['f'], stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeLessThan::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + "<" + stack.fs.pop() + ")");
    }
    
    NodeLessThan* NodeLessThan::clone_impl() const { return new NodeLessThan(*this); }

    NodeLessThan* NodeLessThan::rnd_clone_impl() const { return new NodeLessThan(); }  
}
