/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_variable.h"
			
namespace FT{

    NodeVariable::NodeVariable(const size_t& l, char ntype, std::string n)
    {
        if (n.empty())
	        name = "x_" + std::to_string(l);
        else
            name = n;
	    otype = ntype;
	    complexity = 1;
	    loc = l;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 		
    void NodeVariable::evaluate(const Data& data, Stacks& stack)
    {
        switch(otype)
        {
            case 'b': stack.push<bool>(data.X.row(loc).cast<bool>()); break;
            case 'c': stack.push<int>(data.X.row(loc).cast<int>()); break;
            case 'f': stack.push<double>(data.X.row(loc)); break;
            
        }
    }
#else
    void NodeVariable::evaluate(const Data& data, Stacks& stack)
    {
        if(otype == 'b')
        {
            ArrayXb tmp = data.X.row(loc).cast<bool>();
            GPU_Variable(stack.dev_b, tmp.data(), stack.idx[otype], stack.N);
        }
        else if (otype == 'c')
        {
            ArrayXi tmp = data.X.row(loc).cast<int>();
            GPU_Variable(stack.dev_c, tmp.data(), stack.idx[otype], stack.N);
        }
        else
        {
            ArrayXf tmp = data.X.row(loc).cast<float>() ;
            // std::cout << "NodeVariable:\n stack.dev_f: " << stack.dev_f
            //           << "\ntmp.data(): " << tmp.data() 
            //           << "\ntmp.size(): " << tmp.size()
            //           << "\nstack.idx[otype]: " << stack.idx[otype]
            //           << "\nstack.N: " << stack.N <<"\n";
            GPU_Variable(stack.dev_f, tmp.data(), stack.idx[otype], stack.N);
        }
        
        /*switch(otype)
        {
            case 'b': ArrayXb tmpb = data.X.row(loc).cast<bool>();
                      GPU_Variable(stack.dev_b, tmpb.data(), stack.idx[otype], stack.N);
                      break;
            case 'c': ArrayXi tmpc = data.X.row(loc).cast<int>();
                      GPU_Variable(stack.dev_c, tmpc.data(), stack.idx[otype], stack.N);
                      break;
            case 'f':
            default : ArrayXf tmpf = data.X.row(loc).cast<float>() ;
                      // std::cout << "NodeVariable:\n stack.dev_f: " << stack.dev_f
                      //           << "\ntmp.data(): " << tmp.data() 
                      //           << "\ntmp.size(): " << tmp.size()
                      //           << "\nstack.idx[otype]: " << stack.idx[otype]
                      //           << "\nstack.N: " << stack.N <<"\n";
                      GPU_Variable(stack.dev_f, tmpf.data(), stack.idx[otype], stack.N);
        }*/
    }
#endif

    /// Evaluates the node symbolically
    void NodeVariable::eval_eqn(Stacks& stack)
    {
        switch(otype)
        {
            case 'b' : stack.push<bool>(name); break;
            case 'c' : stack.push<int>(name); break;
            case 'f' : stack.push<double>(name); break;
        }
    }

    NodeVariable* NodeVariable::clone_impl() const { return new NodeVariable(*this); }
      
    // rnd_clone is just clone_impl() for variable, since rand vars not supported
    NodeVariable* NodeVariable::rnd_clone_impl() const { return new NodeVariable(*this); }
}
