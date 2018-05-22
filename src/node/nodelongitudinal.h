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
			
			NodeLongitudinal(std::string n)
			{
                name = "z_"+trim(n);
                
                zName = n;
                    
    			otype = 'z';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['z'] = 0;
    			complexity = 1;
    		}
    		
    		/// Evaluates the node and updates the stack states. 		
			void evaluate(Data& data, Stacks& stack)
		    {
		        stack.z.push(data.Z.at(zName));
		    }

		    /// Evaluates the node symbolically
		    void eval_eqn(Stacks& stack)
		    {
		        stack.zs.push(name);
		    }
        protected:
            NodeLongitudinal* clone_impl() const override { return new NodeLongitudinal(*this); }; 

            NodeLongitudinal* rnd_clone_impl() const override { return clone_impl(); }; 
    };
}

#endif
