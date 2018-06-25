/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_divide.h"
    	  	
namespace FT{

    NodeDivide::NodeDivide(vector<double> W0)
    {
	    name = "/";
	    otype = 'f';
	    arity['f'] = 2;
	    arity['b'] = 0;
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
    void NodeDivide::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        // safe division returns x1/x2 if x2 != 0, and MAX_DBL otherwise               
        stack.f.push( (abs(x2) > NEAR_ZERO ).select((this->W[0] * x1) / (this->W[1] * x2), 
                                                    1.0) ); 
    }
#else
    void NodeDivide::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        GPU_Divide(stack.dev_f, stack.idx[otype], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    void NodeDivide::eval_eqn(Stacks& stack)
    {
        stack.fs.push("(" + stack.fs.pop() + "/" + stack.fs.pop() + ")");            	
    }

    // Might want to check derivative orderings for other 2 arg nodes
    ArrayXd NodeDivide::getDerivative(Trace& stack, int loc) {
        ArrayXd x1 = stack.f[stack.f.size() - 1];
        ArrayXd x2 = stack.f[stack.f.size() - 2];
        switch (loc) {
            case 3: // d/dW[1]
                return limited(-this->W[0] * x1/(x2 * pow(this->W[1], 2)));
            case 2: // d/dW[0]
                return limited(x1/(this->W[1] * x2));
            case 1: // d/dx2 
            {
                /* std::cout << "x1: " << x1.transpose() << "\n"; */
                /* ArrayXd num = -this->W[0] * x1; */
                /* ArrayXd denom = limited(this->W[1] * pow(x2, 2)); */
                /* ArrayXd val = num/denom; */
                return limited((-this->W[0] * x1)/(this->W[1] * pow(x2, 2)));
            }
            case 0: // d/dx1 
            default:
                return limited(this->W[0]/(this->W[1] * x2));
               // return limited(this->W[1]/(this->W[0] * x2));
        } 
    }
    
    NodeDivide* NodeDivide::clone_impl() const { return new NodeDivide(*this); }
      
    NodeDivide* NodeDivide::rnd_clone_impl() const { return new NodeDivide(); } 
}
