/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LONGITUDINAL
#define NODE_LONGITUDINAL

#include "../node.h"

namespace FT{

    namespace Pop{
        namespace Op{
	        class NodeLongitudinal : public Node
	        {
		        public:
			        string zName;
			
			        NodeLongitudinal(std::string n);
            		
            		/// Evaluates the node and updates the state states. 		
			        void evaluate(const Data& data, State& state);

		            /// Evaluates the node symbolically
		            void eval_eqn(State& state);
		            
                protected:
                    NodeLongitudinal* clone_impl() const override;

                    NodeLongitudinal* rnd_clone_impl() const override;
            };
        }
    }
}

#endif
