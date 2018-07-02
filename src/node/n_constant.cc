/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_constant.h"
#include "../error.h"
    		
namespace FT{

    NodeConstant::NodeConstant()
    {
        HANDLE_ERROR_THROW("error in nodeconstant.h : invalid constructor called");
    }

    /// declares a boolean constant
    NodeConstant::NodeConstant(bool& v)
    {
	    name = "k_b";
	    otype = 'b';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    complexity = 1;
	    b_value = v;
    }

    /// declares a double constant
    NodeConstant::NodeConstant(const double& v)
    {
	    name = "k_d";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    complexity = 1;
	    d_value = v;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeConstant::evaluate(Data& data, Stacks& stack)
    {
	    if (otype == 'b')
            stack.b.push(ArrayXb::Constant(data.X.cols(),int(b_value)));
        else 	
            stack.f.push(limited(ArrayXd::Constant(data.X.cols(),d_value)));
    }

    /// Evaluates the node symbolically
    void NodeConstant::eval_eqn(Stacks& stack)
    {
	    if (otype == 'b')
            stack.bs.push(std::to_string(b_value));
        else 	
            stack.fs.push(std::to_string(d_value));
    }
    
    NodeConstant* NodeConstant::clone_impl() const { return new NodeConstant(*this); }
      
    NodeConstant* NodeConstant::rnd_clone_impl() const { return new NodeConstant(); };

}
