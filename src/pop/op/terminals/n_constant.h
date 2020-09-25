/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CONSTANT
#define NODE_CONSTANT

#include "../node.h"

namespace FT{

namespace Pop{
namespace Op{
class NodeConstant : public Node
{
    public:
        
        float d_value;           ///< value, for k and x types
        bool b_value;
        
        NodeConstant();
        
        /// declares a boolean constant
        NodeConstant(bool& v);

        /// declares a float constant
        NodeConstant(const float& v);
        
        /// Evaluates the node and updates the state states. 
        void evaluate(const Data& data, State& state);

        /// Evaluates the node symbolically
        void eval_eqn(State& state);

        // Make the derivative 1
        
    protected:
            NodeConstant* clone_impl() const override;
  
            NodeConstant* rnd_clone_impl() const override;
};
// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeConstant, name, otype, arity, complexity, 
        visits, d_value, b_value)
} // Op
} // Pop
} // FT

#endif
