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
        std::vector<float> W;
        std::vector<float> V;  
    
        virtual ~NodeDx();

        virtual ArrayXf getDerivative(Trace& state, int loc) = 0;
        
        void derivative(vector<ArrayXf>& gradients, Trace& state, int loc);

        void update(vector<ArrayXf>& gradients, Trace& state, float n, float a);

        void print_weight();

        bool isNodeDx();
};
// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeDx, name, otype, arity, complexity, visits, W, V)
}
}
}	

#endif
