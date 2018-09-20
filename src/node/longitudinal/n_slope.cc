/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_slope.h"
#include "../../utils.h"    	

namespace FT{

    NodeSlope::NodeSlope()
    {
        name = "slope";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 4;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeSlope::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        for(int x = 0; x < stack.z.top().first.size(); x++)                    
            tmp(x) = slope(limited(stack.z.top().first[x]), limited(stack.z.top().second[x]));
            
        stack.z.pop();

        stack.push<double>(tmp);
        
    }
#else
    void NodeSlope::evaluate(const Data& data, Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)                    
            stack.f.row(stack.idx['f']) = slope(stack.z.top().first[x], stack.z.top().second[x]);
            
        stack.z.pop();

        
        }
#endif

    /// Evaluates the node symbolically
    void NodeSlope::eval_eqn(Stacks& stack)
    {
        stack.push<double>("slope(" + stack.zs.pop() + ")");
    }
    
    double NodeSlope::slope(const ArrayXd& x, const ArrayXd& y)
    {
        double varx = variance(x);
        if (varx > NEAR_ZERO)
            return covariance(x, y)/varx;
        else
            return 0;
    }

    NodeSlope* NodeSlope::clone_impl() const { return new NodeSlope(*this); }

    NodeSlope* NodeSlope::rnd_clone_impl() const { return new NodeSlope(); } 
}
