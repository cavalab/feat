/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodetanh.h"
    	
namespace FT{

    NodeTanh::NodeTanh(vector<double> W0)
    {
        name = "tanh";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 0;
	    complexity = 3;

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
    void NodeTanh::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(limited(tanh(W[0]*stack.f.pop())));
    }

    /// Evaluates the node symbolically
    void NodeTanh::eval_eqn(Stacks& stack)
    {
        stack.fs.push("tanh(" + stack.fs.pop() + ")");
    }

    ArrayXd NodeTanh::getDerivative(vector<ArrayXd>& stack_f, int loc) {
        ArrayXd numerator;
        ArrayXd denom;
        ArrayXd x = stack_f[stack_f.size()-1];
        switch (loc) {
            case 1: // d/dw0
                numerator = 4 * x * exp(2 * this->W[0] * x);
                denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                // numerator = 4 * x * exp(2 * this->W[0] * x - 1]); 
                // denom = pow(exp(2 * this->W[0] * x) + 1,2);
                return numerator/denom;
            case 0: // d/dx0
            default:
                numerator = 4 * this->W[0] * exp(2 * this->W[0] * x);
                denom = pow(exp(2 * this->W[0] * x) + 1, 2);

                // numerator = 4 * W[0] * exp(2 * W[0] * stack_f[stack_f.size() - 1]);
                // denom = pow(exp(2 * W[0] * stack_f[stack_f.size()-1]),2);
                return numerator/denom;
        } 
    }

    // void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
    //     switch (loc) {
    //         case 0:
    //         default:
    //             numerator = 4 * W[0] * exp(2 * W[0] * stack_f[stack_f.size() - 1]);
    //             denom = pow(exp(2 * W[0] * stack_f[stack_f.size() - 1]),2);
    //             gradients.push_back(numerator/denom);
    //             break;
    //     } 
    // }

    // void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, double n) {
    //     int update_value = 1;
    //     for(auto &grad : gradients) {
    //         update_value *= grad;
    //     }

    //     numerator = 4 * stack_f[stack_f.size() - 1] * exp(2 * W[0] * stack_f[stack_f.size() - 1]); 
    //     denom = pow(exp(2 * W[0] * stack_f[stack_f.size()-1]) + 1,2);
    //     d_w = numerator/denom;
    //     W[0] = W[0] - n/update_value.size() * sum(d_w * update_value);
    // }

    NodeTanh* NodeTanh::clone_impl() const { return new NodeTanh(*this); }
     
    NodeTanh* NodeTanh::rnd_clone_impl() const { return new NodeTanh(); }
}
