/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef STACK_H
#define STACK_H

//#include "node/node.h"
//external includes

namespace FT
{
    template<typename type>
    /*!
     * @class Stack
     * @brief template stack class which holds various stack types for feat 
     */
    class Stack
    {
        private:
            std::vector<type> st;               ///< vector representing the stack
            
        public:
        
            ///< constructor initializing the vector
            Stack()                             
            {
                st = std::vector<type>();
            }
            
            ///< population pushes element at back of vector
            void push(type element){ st.push_back(element); }   
            
            ///< pops element from back of vector and removes it
            type pop()                                          
            {
                type ret = st.back();
                st.pop_back();
                return ret;
            }
            
            ///< returns true or false depending on stack is empty or not
            bool empty(){ return st.empty(); }                  
            
            ///< returns size of stack
            unsigned int size(){ return st.size(); }            
            
            ///< returns top element of stack
            type& top(){ return st.back(); }                    
            
            ///< returns element at particular location in stack
            type& at(int i){ return st.at(i); }                 
            
            ///< clears the stack
            void clear(){ st.clear(); }                         
            
            ///< returns start iterator of stack
            typename vector<type>::iterator begin(){ return st.begin(); }   
            
            ///< returns end iterator of stack
            typename vector<type>::iterator end(){ return st.end(); }       
            
            ///< returns const start iterator of stack
            typename vector<type>::const_iterator begin() const { return st.begin(); }  
            
            ///< returns const iterator of stack
            typename vector<type>::const_iterator end() const { return st.end(); }      
    };
    
    /*!
     * @class Stacks
     * @brief contains various types of stacks actually used by feat
     */
    struct Stacks
    {
        Stack<ArrayXd> f;                   ///< floating node stack
        Stack<ArrayXb> b;                   ///< boolean node stack
        Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;     ///< longitudinal node stack
        Stack<string> fs;                   ///< floating node string stack
        Stack<string> bs;                   ///< boolean node string stack
        Stack<string> zs;                   ///< longitudinal node string stack
        
        ///< checks if arity of node provided satisfies the elements in various value stacks
        bool check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.size() >= arity['f'] && b.size() >= arity['b']);
            else
                return (f.size() >= arity['f'] && b.size() >= arity['b'] 
                        && z.size() >= arity['z']);
        }
        
        ///< checks if arity of node provided satisfies the node names in various string stacks
        bool check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
            else
                return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                        && zs.size() >= arity['z']);
        }
    };
}



#endif
