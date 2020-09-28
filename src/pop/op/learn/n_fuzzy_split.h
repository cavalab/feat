/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_FUZZY_SPLIT
#define NODE_FUZZY_SPLIT

#include "../n_train.h"

namespace FT{

namespace Pop{
namespace Op{
template <class T>
class NodeFuzzySplit : public NodeTrain
{
    public:

        float threshold; 

        NodeFuzzySplit();
        
        /// Uses a heuristic to set a splitting threshold.
        void set_threshold(ArrayXf& x, VectorXf& y, bool classification);

        /// returns the gain of a split 
        float gain(const VectorXf& lsplit, const VectorXf& rsplit, 
                bool classification=false, 
                vector<float> unique_classes = vector<float>());
        
        /// gini impurity of classes in classes
        float gini_impurity_index(const VectorXf& classes,
                 vector<float> uc);
        
        /// Evaluates the node and updates the state states. 
        void evaluate(const Data& data, State& state);            

        /// Evaluates the node symbolically
        void eval_eqn(State& state);
        
    protected:
        NodeFuzzySplit* clone_impl() const override;
        NodeFuzzySplit* rnd_clone_impl() const override;
};
// serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeFuzzySplit<float>, name, otype, arity, complexity, 
        visits, train, threshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NodeFuzzySplit<int>, name, otype, arity, complexity, 
        visits, train, threshold)
}
}
}	

#endif
