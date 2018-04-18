/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MAX
#define NODE_MAX

#include "node.h"

namespace FT{
	class NodeMax : public Node
    {
    	public:
    	
    		NodeMax()
            {
                name = "max";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['l'] = 1;
    			complexity = 1;
    		}
    		
            void evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("max(" + stack.zs.pop() + ")");
            }
        protected:
            NodeMax* clone_impl() const override { return new NodeMax(*this); }; 
    };
#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeMax::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = stack.z.top().first[x].maxCoeff();
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }
#else
    void NodeMax::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = stack.z.top().first[x].maxCoeff();
            
        stack.z.pop();

        
    }
#endif
}	

#endif
