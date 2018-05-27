/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "nodecube.h"

namespace FT{
    		  
    NodeCube::NodeCube(vector<double> W0)
    {
	    name = "cube";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 0;
	    complexity = 33;

        if (W0.empty())
        {
            for (int i = 0; i < arity['f']; i++) {
                W.push_back(r.rnd_dbl());
            }
        }
        else
            W = W0;
    }

    /// Evaluates the node and updates the stack states.  
    void NodeCube::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(pow(this->W[0] * stack.f.pop(),3)));
    }

    /// Evaluates the node symbolically
    void NodeCube::eval_eqn(Stacks& stack)
    {
        stack.fs.push("(" + stack.fs.pop() + "^3)");
    }

    ArrayXd NodeCube::getDerivative(vector<ArrayXd>& stack_f, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return 3 * pow(stack_f[stack_f.size()-1], 3) * pow(this->W[0], 2);
            case 0: // d/dx0
            default:
               return 3 * pow(this->W[0], 3) * pow(stack_f[stack_f.size()-1], 2);
        } 
    }
    
    NodeCube* NodeCube::clone_impl() const { return new NodeCube(*this); }
      
    NodeCube* NodeCube::rnd_clone_impl() const { return new NodeCube(); } 
}
