/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SPLIT
#define NODE_SPLIT

#include "../node.h"

namespace FT{
    template <class T>
	class NodeSplit : public Node
    {
    	public:
    
            double threshold; 

    		NodeSplit();
    		
            /// Uses a heuristic to set a splitting threshold.
            void set_threshold(ArrayXd& x, VectorXd& y, bool classification);

            /// returns the gain of a split 
            double gain(const VectorXd& lsplit, const VectorXd& rsplit, bool classification=false);
            
            /// gini impurity of classes in classes
            double gini_impurity_index(const VectorXd& classes);
            
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);            

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeSplit* clone_impl() const override;
            NodeSplit* rnd_clone_impl() const override;
    };
}	

#endif
