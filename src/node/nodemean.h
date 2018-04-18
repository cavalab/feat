/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEAN
#define NODE_MEAN

#include "node.h"

namespace FT{
	class NodeMean : public Node
    {
    	public:
    	
    		NodeMean()
            {
                name = "mean";
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
                stack.fs.push("mean(" + stack.zs.pop() + ")");
            }
        protected:
            NodeMean* clone_impl() const override { return new NodeMean(*this); }; 
    };
    
#ifndef USE_CUDA 
    /// Evaluates the node and updates the stack states. 
    void NodeMean::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = stack.z.top().first[x].mean();
          
        stack.z.pop();
        
        stack.f.push(tmp);
        
    }
#else
    void NodeMean::evaluate(const MatrixXd& X, const VectorXd& y,
                  const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                  Stacks& stack)
    {
        
        for(unsigned x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = stack.z.top().first[x].mean();
       
        stack.z.pop(); 
    }
#endif

}	

#endif
