#ifndef NODE_TRAIN_H
#define NODE_TRAIN_H

#include "node.h"
// This is a subclass node type for nodes that are trained. Adds a train boolean that can be toggled
// to switch between training and validation.
namespace FT{
namespace Pop{
namespace Op{

class NodeTrain : public Node
{
    public:
        bool train;

        bool isNodeTrain(){ return true; };
};
// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeTrain, name, otype, arity, complexity, 
                                   visits, train)
}
}
}	

#endif
