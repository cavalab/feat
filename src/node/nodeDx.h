#ifndef NODE_DIF_H
#define NODE_DIF_H

#include "node.h"

// There could be a potential problem with the use of loc and how the weights line up with the arguments as the methods I was using are different then the ones feat is using
// Need to remember for implementing auto-backprop that the arguments are in reverse order (top of the stack is the last argument)

namespace FT{
	class NodeDx : public Node
    {
    	public:
    		std::vector<double> W;
            int visits = 0;
    	
    		virtual ~NodeDx(){}

    		virtual ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) = 0;
    		
    		void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                gradients.push_back(getDerivative(stack_f, loc));
            }

            void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, double n, int loc) {
                ArrayXd update_value = ArrayXd::Ones(stack_f[0].size());
                for(ArrayXd g : gradients) {
                    update_value *= g;
                }

                // Update all weights
                vector<double> W_temp(W);	// Have to use temporary weights so as not to compute updates with updated weights
                for (int i = 0; i < arity['f']; ++i) {
                	ArrayXd d_w = getDerivative(stack_f, arity['f'] + i);
                	W_temp[i] = W[i] - n/update_value.size() * (d_w * update_value).sum();
                }
                this->W = W_temp;
            }
    };
}	

#endif
