/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_relu.h"

namespace FT{
 	
    NodeRelu::NodeRelu(vector<double> W0)
    {
	    name = "relu";
	    otype = 'f';
	    arity['f'] = 1;
	    complexity = 2;

        if (W0.empty())
        {
            for (int i = 0; i < arity['f']; i++) {
                W.push_back(r.rnd_dbl());
            }
        }
        else
            W = W0;
    }

#ifndef USE_CUDA 
    /// Evaluates the node and updates the stack states. 
    void NodeRelu::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd x = stack.f.pop();
        ArrayXd res = (W[0] * x > 0).select(W[0]*x, ArrayXd::Zero(x.size())+0.01); 
        stack.f.push(res);
    }
#else
    /// Evaluates the node and updates the stack states. 
    void NodeRelu::evaluate(const Data& data, Stacks& stack)
    {
        GPU_Relu(stack.dev_f, stack.idx[otype], stack.N, (float)W[0]);
    }
#endif
    /// Evaluates the node symbolically
    void NodeRelu::eval_eqn(Stacks& stack)
    {
        stack.fs.push("relu("+ stack.fs.pop() +")");         	
    }

    ArrayXd NodeRelu::getDerivative(Trace& stack, int loc) {

        ArrayXd x = stack.f[stack.f.size()-1];
        switch (loc) {
            case 1: // d/dW
                return (x>0).select(x,ArrayXd::Zero(x.size())+0.01);
            case 0: // d/dx
            default:
                return (x>0).select(W[0],ArrayXd::Zero(x.size())+0.01);
        } 
    }
    
    NodeRelu* NodeRelu::clone_impl() const { return new NodeRelu(*this); }

    NodeRelu* NodeRelu::rnd_clone_impl() const { return new NodeRelu(); }  
}
