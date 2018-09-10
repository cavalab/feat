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
	    complexity = 1;
	    b_value = v;
    }

    /// declares a double constant
    NodeConstant::NodeConstant(const double& v)
    {
	    name = "k_d";
	    otype = 'f';
	    complexity = 1;
	    d_value = v;
    }

#ifndef USE_CUDA    
    /// Evaluates the node and updates the stack states. 
    void NodeConstant::evaluate(const Data& data, Stacks& stack)
    {
	    if (otype == 'b')
            stack.push<bool>(ArrayXb::Constant(data.X.cols(),int(b_value)));
        else 	
            stack.push<double>(limited(ArrayXd::Constant(data.X.cols(),d_value)));
    }
#else
    void NodeConstant::evaluate(const Data& data, Stacks& stack)
    {
        if (otype == 'b')
        {
            GPU_Constant(stack.dev_b, b_value, stack.idx['b'], stack.N);
        }
        else
        {
            GPU_Constant(stack.dev_f, d_value, stack.idx['f'], stack.N);
        }

    }
#endif

    /// Evaluates the node symbolically
    void NodeConstant::eval_eqn(Stacks& stack)
    {
	    if (otype == 'b')
            stack.push<bool>(std::to_string(b_value));
        else 	
            stack.push<double>(std::to_string(d_value));
    }
    
    NodeConstant* NodeConstant::clone_impl() const { return new NodeConstant(*this); }
      
    NodeConstant* NodeConstant::rnd_clone_impl() const { return new NodeConstant(); };

}
