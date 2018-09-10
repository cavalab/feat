/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_geq.h"
    	
namespace FT{

    NodeGEQ::NodeGEQ()
    {
	    name = ">=";
	    otype = 'b';
	    arity['f'] = 2;
	    complexity = 2;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeGEQ::evaluate(const Data& data, Stacks& stack)
    {
	    ArrayXd x1 = stack.pop<double>();
        ArrayXd x2 = stack.pop<double>();
        stack.push<bool>(x1 >= x2);
    }
#else
    void NodeGEQ::evaluate(const Data& data, Stacks& stack)
    {
        GPU_GEQ(stack.dev_f, stack.dev_b, stack.idx['f'], stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeGEQ::eval_eqn(Stacks& stack)
    {
        stack.push<bool>("(" + stack.popStr<double>() + ">=" + stack.popStr<double>() + ")");
    }
    
    NodeGEQ* NodeGEQ::clone_impl() const { return new NodeGEQ(*this); }
      
    NodeGEQ* NodeGEQ::rnd_clone_impl() const { return new NodeGEQ(); } 
}
