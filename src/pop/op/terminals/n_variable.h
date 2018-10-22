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
			            		
            		/// Evaluates the node and updates the stack states. 		
			        void evaluate(const Data& data, Stacks& stack);

		            /// Evaluates the node symbolically
		            void eval_eqn(Stacks& stack);
		            
	            protected:
                    NodeVariable* clone_impl() const override;  
                    // rnd_clone is just clone_impl() for variable, since rand vars not supported
                    NodeVariable* rnd_clone_impl() const override;  
            };
        }
    }
}

#endif
