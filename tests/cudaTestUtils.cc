#include "cudaTestUtils.h"

#ifdef USE_CUDA
std::map<char, size_t> get_max_state_size(NodeVector &nodes)
{
    // max state size is calculated using node arities
    std::map<char, size_t> state_size;
    std::map<char, size_t> max_state_size;
    state_size['f'] = 0;
    state_size['c'] = 0;
    state_size['b'] = 0; 
    max_state_size['f'] = 0;
    max_state_size['c'] = 0;
    max_state_size['b'] = 0;

    for (const auto& n : nodes)   
    {   	
        ++state_size[n->otype];

        if ( max_state_size[n->otype] < state_size[n->otype])
            max_state_size[n->otype] = state_size[n->otype];

        for (const auto& a : n->arity)
            state_size[a.first] -= a.second;       
    }	
    return max_state_size;
}

bool isValidProgram(NodeVector& program, unsigned num_features)
{
    //checks whether program fulfills all its arities.
    MatrixXf X = MatrixXf::Zero(num_features,2); 
    VectorXf y = VectorXf::Zero(2);
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 
    
    Data data(X, y, z);
    
    State state;
    
    std::map<char, size_t> state_size = get_max_state_size(program);
    choose_gpu();        
        
    state.allocate(state_size,data.X.cols());        

    for (const auto& n : program)
    {
    	if(state.check(n->arity))
    	{
            n->evaluate(data, state);
            state.update_idx(n->otype, n->arity); 
        }
        else
            return false;   
    }
    
    return true;
}
#endif
