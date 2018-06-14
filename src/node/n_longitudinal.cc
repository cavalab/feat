/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_longitudinal.h"
#include "../utils.h"
	
namespace FT{
	
    NodeLongitudinal::NodeLongitudinal(std::string n)
    {
        name = "z_"+trim(n);
        
        zName = n;
            
	    otype = 'z';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 0;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 		
    void NodeLongitudinal::evaluate(Data& data, Stacks& stack)
    {
        stack.z.push(data.Z.at(zName));
    }

    /// Evaluates the node symbolically
    void NodeLongitudinal::eval_eqn(Stacks& stack)
    {
        stack.zs.push(name);
    }
    
    NodeLongitudinal* NodeLongitudinal::clone_impl() const { return new NodeLongitudinal(*this); }

    NodeLongitudinal* NodeLongitudinal::rnd_clone_impl() const { return clone_impl(); }

}
