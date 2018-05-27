/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "stack.h"

namespace FT
{

    template<typename type>
    Stack<type>::Stack()                             
    {
        st = std::vector<type>();
    }
    
    ///< population pushes element at back of vector
    template<typename type>
    void Stack<type>::push(type element){ st.push_back(element); }   
    
    ///< pops element from back of vector and removes it
    template<typename type>
    type Stack<type>::pop()                                          
    {
        type ret = st.back();
        st.pop_back();
        return ret;
    }
    
    ///< returns true or false depending on stack is empty or not
    template<typename type>
    bool Stack<type>::empty(){ return st.empty(); }                  
    
    ///< returns size of stack
    template<typename type>
    unsigned int Stack<type>::size(){ return st.size(); }            
    
    ///< returns top element of stack
    template<typename type>
    type& Stack<type>::top(){ return st.back(); }                    
    
    ///< returns element at particular location in stack
    template<typename type>
    type& Stack<type>::at(int i){ return st.at(i); }                 
    
    ///< clears the stack
    template<typename type>
    void Stack<type>::clear(){ st.clear(); }                         
    
    ///< returns start iterator of stack
    template<typename type>
    typename vector<type>::iterator Stack<type>::begin(){ return st.begin(); }   
    
    ///< returns end iterator of stack
    template<typename type>
    typename vector<type>::iterator Stack<type>::end(){ return st.end(); }       
    
    ///< returns const start iterator of stack
    template<typename type>
    typename vector<type>::const_iterator Stack<type>::begin() const { return st.begin(); }  
    
    ///< returns const iterator of stack
    template<typename type>
    typename vector<type>::const_iterator Stack<type>::end() const { return st.end(); }      

    bool Stacks::check(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (f.size() >= arity['f'] && b.size() >= arity['b']);
        else
            return (f.size() >= arity['f'] && b.size() >= arity['b'] 
                    && z.size() >= arity['z']);
    }
    
    ///< checks if arity of node provided satisfies the node names in various string stacks
    bool Stacks::check_s(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
        else
            return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                    && zs.size() >= arity['z']);
    }
}

