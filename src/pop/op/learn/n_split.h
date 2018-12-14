/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SPLIT
#define NODE_SPLIT

#include "../n_train.h"

namespace FT{

    namespace Pop{
        namespace Op{
            template <class T>
	        class NodeSplit : public NodeTrain
            {
            	public:
            
                    float threshold; 

            		NodeSplit();
            		
                    /// Uses a heuristic to set a splitting threshold.
                    void set_threshold(ArrayXf& x, VectorXf& y, bool classification);

                    /// returns the gain of a split 
                    float gain(const VectorXf& lsplit, const VectorXf& rsplit, bool classification=false);
                    
                    /// gini impurity of classes in classes
                    float gini_impurity_index(const VectorXf& classes);
                    
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);            

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                    
                protected:
                    NodeSplit* clone_impl() const override;
                    NodeSplit* rnd_clone_impl() const override;
            };
        }
    }
}	

#endif
