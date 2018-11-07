#ifndef NODE_DIF_H
#define NODE_DIF_H

#include "node.h"

// There could be a potential problem with the use of loc and how the weights line up with the arguments as the methods I was using are different then the ones feat is using
// Need to remember for implementing auto-backprop that the arguments are in reverse order (top of the state is the last argument)

namespace FT{

    namespace Pop{
        namespace Op{

            class NodeDx : public Node
            {
            	public:
            		std::vector<double> W;
                    std::vector<double> V;  
            	
            		virtual ~NodeDx();

            		virtual ArrayXd getDerivative(Trace& state, int loc) = 0;
            		
            		void derivative(vector<ArrayXd>& gradients, Trace& state, int loc);

                    void update(vector<ArrayXd>& gradients, Trace& state, double n, double a);

                    void print_weight();

                    bool isNodeDx();
            };
        }
    }
}	

#endif
