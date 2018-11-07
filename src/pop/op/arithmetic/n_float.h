/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_FLOAT
#define NODE_FLOAT

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
            template <class T>
	        class NodeFloat : public Node
            {
            	public:
            	
            		NodeFloat();
            		
                    /// Evaluates the node and updates the state states. 
                    void evaluate(const Data& data, State& state);

                    /// Evaluates the node symbolically
                    void eval_eqn(State& state);
                   
                    /// Determines whether to convert categorical or boolean inputs
                    bool isCategorical;

                protected:
                    NodeFloat* clone_impl() const override;

                    NodeFloat* rnd_clone_impl() const override;
            };
        }
    }
    
}	

#endif
