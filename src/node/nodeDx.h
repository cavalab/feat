#ifndef NODE_DIF_H
#define NODE_DIF_H

#include "node.h"

// There could be a potential problem with the use of loc and how the weights line up with the arguments as the methods I was using are different then the ones feat is using
// Need to remember for implementing auto-backprop that the arguments are in reverse order (top of the stack is the last argument)

namespace FT{
    
    extern Rnd r;   // forward declaration of random number generator

    class NodeDx : public Node
    {
    	public:
    		std::vector<double> W;
            
    	
    		virtual ~NodeDx(){}

    		virtual ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) = 0;
    		
    		void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) 
            {
                gradients.push_back(getDerivative(stack_f, loc));
            }

            void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, double n) 
            {
                std::cout << "***************************\n";
                std::cout << "Updating " << this->name << "\n";
                ArrayXd update_value = ArrayXd::Ones(stack_f[0].size());
                for(ArrayXd g : gradients) {
                    std::cout << "Using gradient: " << g << "\n";
                    update_value *= g;
                }

                // Update all weights
                // std::cout << "Update value: " << update_value << "\n";
                // std::cout << "Input: " << stack_f[stack_f.size() - 1] << "\n";
                vector<double> W_temp(W);	// Have to use temporary weights so as not to compute updates with updated weights
                for (int i = 0; i < arity['f']; ++i) {
                	ArrayXd d_w = getDerivative(stack_f, arity['f'] + i);
                    // std::cout << "Derivative: " << d_w << "\n";
                	W_temp[i] = W[i] - n/update_value.size() * (d_w * update_value).sum();
                    // std::cout << "Updated with " << (d_w * update_value).sum() << "\n";
                }
                this->W = W_temp;
                std::cout << "Updated\n";
                std::cout << "***************************\n";
                // print_weight();
            }

            void print_weight()
            {
                std::cout << this->name << "|W has value";
                for (int i = 0; i < this->arity['f']; i++) {
                    std::cout << " " << this->W[i];
                }
                std::cout << "\n";
            }
    };
}	

#endif
