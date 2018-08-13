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
            std::vector<double> V;  
    	
    		virtual ~NodeDx();

    		virtual ArrayXd getDerivative(Trace& stack, int loc) = 0;
    		
    		void derivative(vector<ArrayXd>& gradients, Trace& stack, int loc);

            void update(vector<ArrayXd>& gradients, Trace& stack, double n, double a);

            void print_weight();

            bool isNodeDx();
    };
}	

#endif
