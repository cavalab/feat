/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_constant.h"
    		
namespace FT{

namespace Pop{
namespace Op{

/// declares a boolean constant
NodeConstant::NodeConstant(bool& v)
{
    name = "constant_b";
    otype = 'b';
    complexity = 1;
    b_value = v;
}

/// declares a float constant
NodeConstant::NodeConstant(const float& v)
{
    name = "constant_d";
    otype = 'f';
    complexity = 1;
    d_value = v;
}

#ifndef USE_CUDA    
/// Evaluates the node and updates the state states. 
void NodeConstant::evaluate(const Data& data, State& state)
{
    if (otype == 'b')
        state.push<bool>(ArrayXb::Constant(data.X.cols(),int(b_value)));
    else 	
        state.push<float>(limited(ArrayXf::Constant(data.X.cols(),d_value)));
}
#else
void NodeConstant::evaluate(const Data& data, State& state)
{
    if (otype == 'b')
    {
        GPU_Constant(state.dev_b, b_value, state.idx['b'], state.N);
    }
    else
    {
        GPU_Constant(state.dev_f, d_value, state.idx['f'], state.N);
    }

}
#endif

/// Evaluates the node symbolically
void NodeConstant::eval_eqn(State& state)
{
    if (otype == 'b')
        state.push<bool>(std::to_string(b_value));
    else 	
        state.push<float>(std::to_string(d_value));
}

NodeConstant* NodeConstant::clone_impl() const { return new NodeConstant(*this); }
  
NodeConstant* NodeConstant::rnd_clone_impl() const { return new NodeConstant(r()); }

}
}
}
