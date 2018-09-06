/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    template <>
    NodeFloat<double>::NodeFloat()
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
        otype = 'c';
    
        arity['b'] = 1;
        complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    template <>
    void NodeFloat<double>::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<double>(stack.pop<bool>().cast<double>());
    }
    
    template <>
    void NodeFloat<int>::evaluate(const Data& data, Stacks& stack)
    {
        stack.push<int>(stack.pop<bool>().cast<int>());
    }

    /// Evaluates the node symbolically
    template <>
    void NodeFloat<double>::eval_eqn(Stacks& stack)
    {
        stack.push<double>("f(" + stack.popStr<bool>() + ")");
    }
    
    template <>
    void NodeFloat<int>::eval_eqn(Stacks& stack)
    {
        stack.push<int>("f_c(" + stack.popStr<bool>() + ")");
    }
    
    template <class T>
    NodeFloat<T>* NodeFloat<T>::clone_impl() const { return new NodeFloat<T>(*this); }

    template <class T>
    NodeFloat<T>* NodeFloat<T>::rnd_clone_impl() const { return new NodeFloat<T>(); }  
}
