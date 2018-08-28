/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_log.h"

namespace FT{

    NodeLog::NodeLog(vector<double> W0)
    {
	    name = "log";
	    otype = 'f';
	    arity['f'] = 1;
	    complexity = 4;

        if (W0.empty())
        {
            for (int i = 0; i < arity['f']; i++) {
                W.push_back(r.rnd_dbl());
            }
        }
        else
            W = W0;
    }

    /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
    void NodeLog::evaluate(Data& data, Stacks& stack)
    {
	    ArrayXd x = stack.pop<double>();
        stack.push<double>( (abs(x) > NEAR_ZERO).select(log(abs(W[0] * x)),MIN_DBL));
    }

    /// Evaluates the node symbolically
    void NodeLog::eval_eqn(Stacks& stack)
    {
        stack.push<double>("log(" + stack.popStr<double>() + ")");
    }

    ArrayXd NodeLog::getDerivative(Trace& stack, int loc)
    {
        ArrayXd& x = stack.get<double>()[stack.size<double>()-1];
        
        switch (loc) {
            case 1: // d/dw0
                return limited(1/(W[0] * ArrayXd::Ones(x.size())));
            case 0: // d/dx0
            default:
                return limited(1/x);
        } 
    }
    
    NodeLog* NodeLog::clone_impl() const { return new NodeLog(*this); }

    NodeLog* NodeLog::rnd_clone_impl() const { return new NodeLog(); }  
}
