#include "nodeDx.h"
    		
namespace FT{

    void NodeDx::derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) 
    {
        gradients.push_back(getDerivative(stack_f, loc));
    }

    void NodeDx::update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, double n, double a) 
    {
        /*! update weights via gradient descent + momentum
         * @params n : learning rate
         * @params a : momentum
         * v(t+1) = a * v(t) - n * gradient
         * w(t+1) = w(t) + v(t+1)
         * */
        if (V.empty())  // first time through, V is zeros
        {
            for (const auto& w : W)
                V.push_back(0.0);
        }
        /* std::cout << "***************************\n"; */
        /* std::cout << "Updating " << this->name << "\n"; */
        ArrayXd update_value = ArrayXd::Ones(stack_f[0].size());
        for(const ArrayXd& g : gradients) {
            /* std::cout << "Using gradient: " << g << "\n"; */
            update_value *= g;
        }

        // Update all weights
        // std::cout << "Update value: " << update_value << "\n";
        // std::cout << "Input: " << stack_f[stack_f.size() - 1] << "\n";
        /* vector<double> W_temp(W); */	
        vector<double> V_temp(V);
        // Have to use temporary weights so as not to compute updates with updated weights
        for (int i = 0; i < arity['f']; ++i) {
        	ArrayXd d_w = getDerivative(stack_f, arity['f'] + i);
            /* std::cout << "Derivative: " << (d_w*update_value).sum() << "\n"; */
            /* std::cout << "V[i]: " << V[i] << "\n"; */
            V_temp[i] = a * V[i] - n/update_value.size() * (d_w * update_value).sum();
            /* std::cout << "V_temp: " << V_temp[i] << "\n"; */
            /* dW[i] = a*dW[i] + (1-a)*( n/update_value.size() * (d_w * update_value).sum()); */
        	/* W_temp[i] = W[i] + dW_temp[i]; */
        	/* W_temp[i] = W[i] - n/update_value.size() * (d_w * update_value).sum(); */
            // std::cout << "Updated with " << (d_w * update_value).sum() << "\n";
        }
        for (int i = 0; i < W.size(); ++i)
        {
            if (std::isfinite(V_temp[i]) && !std::isnan(V_temp[i]))
            {
                this->W[i] += V_temp[i];
                this->V[i] = V_temp[i];
            }
        }

        /* std::cout << "Updated\n"; */
        /* std::cout << "***************************\n"; */
        // print_weight();
    }

    void NodeDx::print_weight()
    {
        std::cout << this->name << "|W has value";
        for (int i = 0; i < this->arity['f']; i++) {
            std::cout << " " << this->W[i];
        }
        std::cout << "\n";
    }

    bool NodeDx::isNodeDx(){ return true;}
}
