#include "cudaTestUtils.h"

#ifdef USE_CUDA
std::map<char, size_t> get_max_stack_size(NodeVector &nodes)
{
    // max stack size is calculated using node arities
    std::map<char, size_t> stack_size;
    std::map<char, size_t> max_stack_size;
    stack_size['f'] = 0;
    stack_size['c'] = 0;
    stack_size['b'] = 0; 
    max_stack_size['f'] = 0;
    max_stack_size['c'] = 0;
    max_stack_size['b'] = 0;

    for (const auto& n : nodes)   
    {   	
        ++stack_size[n->otype];

        if ( max_stack_size[n->otype] < stack_size[n->otype])
            max_stack_size[n->otype] = stack_size[n->otype];

        for (const auto& a : n->arity)
            stack_size[a.first] -= a.second;       
    }	
    return max_stack_size;
}

bool isValidProgram(NodeVector& program, unsigned num_features)
{
    //checks whether program fulfills all its arities.
    MatrixXd X = MatrixXd::Zero(num_features,2); 
    VectorXd y = VectorXd::Zero(2);
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z; 
    
    Data data(X, y, z);
    
    Stacks stack;
    
    std::map<char, size_t> stack_size = get_max_stack_size(program);
    choose_gpu();        
        
    stack.allocate(stack_size,data.X.cols());        

    for (const auto& n : program)
    {
    	if(stack.check(n->arity))
    	{
            n->evaluate(data, stack);
            stack.update_idx(n->otype, n->arity); 
        }
        else
            return false;   
    }
    
    return true;
}
#endif
