/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LONGITUDINAL
#define NODE_LONGITUDINAL

#include "node.h"

namespace FT{
	class NodeLongitudinal : public Node
	{
		public:
			string zName;
			
			NodeLongitudinal(std::string n);
    		
    		/// Evaluates the node and updates the stack states. 		
			void evaluate(Data& data, Stacks& stack);

		    /// Evaluates the node symbolically
		    void eval_eqn(Stacks& stack);
		    
        protected:
            NodeLongitudinal* clone_impl() const override;

            NodeLongitudinal* rnd_clone_impl() const override;
    };
}

#endif
