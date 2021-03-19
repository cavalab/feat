/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_VARIABLE
#define NODE_VARIABLE

#include "../node.h"

namespace FT{

namespace Pop{
namespace Op{
template <class T>
class NodeVariable : public Node
{
    public:
        size_t loc;             ///< column location in X, for x types

        NodeVariable(const size_t& l, char ntype = 'f', std::string n="");
        NodeVariable();
                    
        /// Evaluates the node and updates the state states. 		
        void evaluate(const Data& data, State& state);

        /// Evaluates the node symbolically
        void eval_eqn(State& state);

    protected:
        NodeVariable* clone_impl() const override;  
        // rnd_clone is just clone_impl() for variable, since rand vars not supported
        NodeVariable* rnd_clone_impl() const override;  
};

// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeVariable<float>, name, otype, arity, complexity, 
        visits, loc, variable_name)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeVariable<bool>, name, otype, arity, complexity, 
        visits, loc, variable_name)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeVariable<int>, name, otype, arity, complexity, 
        visits, loc, variable_name)
} // Op
} // Pop
} // FT

#endif
