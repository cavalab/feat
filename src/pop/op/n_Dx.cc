#include "n_Dx.h"
    		
namespace FT{

    namespace Pop{
        namespace Op{

            NodeDx::~NodeDx(){}
	
	        void NodeDx::derivative(vector<ArrayXf>& gradients, Trace& state, int loc) 
            {
                gradients.push_back(getDerivative(state, loc));
            }

            void NodeDx::update(vector<ArrayXf>& gradients, Trace& state, float n, float a) 
            {
                /*! update weights via gradient descent + momentum
                 * @param n : learning rate
                 * @param a : momentum
                 * v(t+1) = a * v(t) - n * gradient
                 * w(t+1) = w(t) + v(t+1)
                 * */
                if (V.empty())  // first time through, V is zeros
                {
                    for (const auto& w : W)
                        V.push_back(0.0);
                }
        //        std::cout << "***************************\n";
        //        std::cout << "Updating " << this->name << "\n";
                ArrayXf update_value = ArrayXf::Ones(state.f[0].size());
                for(const ArrayXf& g : gradients) {
                    /* std::cout << "Using gradient: " << g << "\n"; */
                    update_value *= g;
                }

                // Update all weights
                 /* std::cout << "Update value: " << update_value << "\n"; */
                 /* std::cout << "Input: " << state.f[state.f.size() - 1] << "\n"; */
                 /* std::cout << "Input: " << state.f[state.f.size() - 2] << "\n"; */
                 vector<float> W_temp(W);
                vector<float> V_temp(V);
                
        //        cout << "*****n value is "<< n<<"\n";
                // Have to use temporary weights so as not to compute updates with updated weights
                for (int i = 0; i < arity['f']; ++i) {
                	ArrayXf d_w = getDerivative(state, arity['f'] + i);
                    /* std::cout << "Derivative: " << (d_w*update_value).sum() << "\n"; */
                    /* std::cout << "V[i]: " << V[i] << "\n"; */
        //            V_temp[i] = a * V[i] - n/update_value.size() * (d_w * update_value).sum();
                    V_temp[i] = - n/update_value.size() * (d_w * update_value).sum();
                    /* std::cout << "V_temp: " << V_temp[i] << "\n"; */
                    /* dW[i] = a*dW[i] + (1-a)*( n/update_value.size() * (d_w * update_value).sum()); */
                	/* W_temp[i] = W[i] + dW_temp[i]; */
                	/* W_temp[i] = W[i] - n/update_value.size() * (d_w * update_value).sum(); */
                    /* std::cout << "Updated with " << (d_w * update_value).sum() << "\n"; */
                }
                for (int i = 0; i < W.size(); ++i)
                {
                    if (std::isfinite(V_temp[i]) && !std::isnan(V_temp[i]))
                    {
                        this->W[i] += V_temp[i];
                        this->V[i] = V_temp[i];
                    }
                }

        //        std::cout << "Updated\n";
        //        std::cout << "***************************\n";
        //        print_weight();
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
    }
}
