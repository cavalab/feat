/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CONSTANT
#define NODE_CONSTANT

#include "../node.h"

namespace FT{
	class NodeConstant : public Node
    {
    	public:
    		
    		double d_value;           ///< value, for k and x types
    		bool b_value;
    		
    		NodeConstant();
    		
            /// declares a boolean constant
    		NodeConstant(bool& v);

            /// declares a double constant
    		NodeConstant(const double& v);
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);

            // Make the derivative 1
    		
        protected:
                NodeConstant* clone_impl() const override;
      
                NodeConstant* rnd_clone_impl() const override;
    };
}	

#endif
