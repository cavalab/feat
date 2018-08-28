/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_step.h"


namespace FT{

    NodeStep::NodeStep()
    {
        name = "step";
	    otype = 'f';
	    arity['f'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeStep::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXd x = stack.pop<double>();
	    ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), ArrayXd::Zero(x.size())); 
        stack.push<double>(res);
    }

    /// Evaluates the node symbolically
    void NodeStep::eval_eqn(Stacks& stack)
    {
        stack.push<double>("step("+ stack.popStr<double>() +")");
    }

    NodeStep* NodeStep::clone_impl() const { return new NodeStep(*this); }

    NodeStep* NodeStep::rnd_clone_impl() const { return new NodeStep(); }  
}
