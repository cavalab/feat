/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SLOPE
#define NODE_SLOPE

#include "node.h"

namespace FT{
	class NodeSlope : public Node
    {
    	public:
    	
    		NodeSlope()
            {
                name = "slope";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['z'] = 1;
    			complexity = 4;
    		}
    		
            void evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("slope(" + stack.zs.pop() + ")");
            }
        protected:
            NodeSlope* clone_impl() const override { return new NodeSlope(*this); }; 
    };
#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeSlope::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)                    
            tmp(x) = slope(stack.z.top().first[x], stack.z.top().second[x]);
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }
#else
void NodeSlope::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)                    
            stack.f.row(stack.idx['f']) = slope(stack.z.top().first[x], stack.z.top().second[x]);
            
        stack.z.pop();

        
        }
#endif
}	

#endif
