/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_variable.h"
			
namespace FT{

    NodeVariable::NodeVariable(const size_t& l, char ntype, std::string n)
    {
        if (n.empty())
	        name = "x_" + std::to_string(l);
        else
            name = n;
	    otype = ntype;
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['c'] = 0;
	    complexity = 1;
	    loc = l;
    }

    /// Evaluates the node and updates the stack states. 		
    void NodeVariable::evaluate(Data& data, Stacks& stack)
    {
        switch(otype)
        {
            case 'b': stack.b.push(data.X.row(loc).cast<bool>()); break;
            case 'c': stack.c.push(data.X.row(loc)); break;
            case 'f': stack.f.push(data.X.row(loc)); break;
            
        }
    }

    /// Evaluates the node symbolically
    void NodeVariable::eval_eqn(Stacks& stack)
    {
        switch(otype)
        {
            case 'b' : stack.bs.push(name); break;
            case 'c' : stack.cs.push(name); break;
            case 'f' : stack.fs.push(name); break;
        }
    }

    NodeVariable* NodeVariable::clone_impl() const { return new NodeVariable(*this); }
      
    // rnd_clone is just clone_impl() for variable, since rand vars not supported
    NodeVariable* NodeVariable::rnd_clone_impl() const { return new NodeVariable(*this); }
}
