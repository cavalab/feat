/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    template <>
    NodeFloat<bool>::NodeFloat()
    {
        name = "f";
        otype = 'f';
        arity['b'] = 1;
        complexity = 1;
    }
    
    template <>
    NodeFloat<int>::NodeFloat()
    {
        name = "f_c";
        otype = 'f';
        arity['c'] = 1;
        complexity = 1;
    }

#ifndef USE_CUDA 
    /// Evaluates the node and updates the stack states. 
    template <class T>
    void NodeFloat<T>::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(stack.pop<T>().template cast<double>());
    }
#else
    template <class T>
    void NodeFloat<T>::evaluate(const Data& data, Stacks& stack)
    {
        if(arity['b'])
            GPU_Float(stack.dev_f, stack.dev_b, stack.idx[otype], stack.idx['b'], stack.N);
        else
            GPU_Float(stack.dev_f, stack.dev_c, stack.idx[otype], stack.idx['c'], stack.N);
    }
#endif

    /// Evaluates the node symbolically
    template <class T>
    void NodeFloat<T>::eval_eqn(Stacks& stack)
    {
        stack.push<double>("float(" + stack.popStr<T>() + ")");
    }
    
    template <class T>
    NodeFloat<T>* NodeFloat<T>::clone_impl() const { return new NodeFloat<T>(*this); }

    template <class T>
    NodeFloat<T>* NodeFloat<T>::rnd_clone_impl() const { return new NodeFloat<T>(*this); }  
}
