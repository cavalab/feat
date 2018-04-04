#ifndef NODE_DIF_H
#define NODE_DIF_H

#include "node.h"

// There could be a potential problem with the use of loc and how the weights line up with the arguments as the methods I was using are different then the ones feat is using
// Need to remember for implementing auto-backprop that the arguments are in reverse order (top of the stack is the last argument)

namespace FT{
	class NodeDx : public Node
    {
    	public:
    		vector<int> W;
    	
    		virtual ~NodeDx(){}

    		virtual ArrayXd getDerivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) = 0;
    		
    		void derivative(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                gradients.push_back(getDerivative(gradients, stack_f, loc));
            }

            void update(vector<ArrayXd>& gradients, vector<ArrayXd>& stack_f, int loc) {
                update_value = 1
                for(auto g : gradients) {
                    update_value *= g;
                }

                // Update all weights
                W_temp = W[:];	// Have to use temporary weights so as not to compute updates with updated weights
                for (int i = 0; i < arity['f']; ++i) {
                	d_w = getDerivative(gradients, stack_f, arity['f'] + i);
                	W_temp[i] = W - n/update_value.size * sum(d_w * update_value);
                }
                W = W_temp[:];
            }
    };
}	

#endif