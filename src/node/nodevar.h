/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_VARIANCE
#define NODE_VARIANCE

#include "node.h"

namespace FT{
	class NodeVar : public Node
    {
    	public:
    	
    		NodeVar()
            {
                name = "variance";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['z'] = 1;
    			complexity = 1;
    		}
    		
            void evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("variance(" + stack.zs.pop() + ")");
            }
        protected:
            NodeVar* clone_impl() const override { return new NodeVar(*this); }; 
    };
#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeVar::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        double mean;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = variance(stack.z.top().first[x]);
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }
#else
void NodeVar::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = variance(stack.z.top().first[x]);
            
        stack.z.pop();
        
    }
#endif
}	

#endif
