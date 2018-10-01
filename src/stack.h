/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef STACK_H
#define STACK_H

#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <iostream>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXi;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;

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
            typename vector<type>::const_iterator begin() const{ return st.begin(); }
            
            ///< returns const iterator of stack
            typename vector<type>::const_iterator end() const{ return st.end(); }
            
            ~Stack(){}
    };
    
    /*!
     * @class Stacks
     * @brief contains various types of stacks actually used by feat
     */
    struct Stacks
    {
        Stack<ArrayXd> f;                   ///< floating node stack
        Stack<ArrayXb> b;                   ///< boolean node stack
        Stack<ArrayXi> c;                   ///<categorical stack
        Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;     ///< longitudinal node stack
        Stack<string> fs;                   ///< floating node string stack
        Stack<string> bs;                   ///< boolean node string stack
        Stack<string> cs;                   ///< categorical node string stack
        Stack<string> zs;                   ///< longitudinal node string stack
        
        ///< checks if arity of node provided satisfies the elements in various value stacks
        bool check(std::map<char, unsigned int> &arity);
        
        ///< checks if arity of node provided satisfies the node names in various string stacks
        bool check_s(std::map<char, unsigned int> &arity);
        
        template <typename T> inline Stack<Eigen::Array<T,Eigen::Dynamic,1> >& get()
        {
            return get<Eigen::Array<T,Eigen::Dynamic,1> >();
        }
        
        template <typename T> void push(Eigen::Array<T,Eigen::Dynamic,1>  value)
        {
            get<T>().push(value);
        }

        template <typename T> Eigen::Array<T,Eigen::Dynamic,1>  pop()
        {
            return get<T>().pop();
        }
        
        template <typename T> inline Stack<string>& getStr()
        {
            return getStr<T>();
        }
        
        template <typename T> void push(string value)
        {
            getStr<T>().push(value);
        }
        
        template <typename T> string popStr()
        {
            return getStr<T>().pop();
        }
        
        template <typename T> unsigned int size()
        {
            return get<T>().size();
        }
        
    };
    
    template <> inline Stack<ArrayXd>& Stacks::get(){ return f; }
        
    template <> inline Stack<ArrayXb>& Stacks::get(){ return b; }
    
    template <> inline Stack<ArrayXi>& Stacks::get(){ return c; }
    
    template <> inline Stack<string>& Stacks::getStr<double>(){ return fs; }
        
    template <> inline Stack<string>& Stacks::getStr<bool>(){ return bs; }
    
    template <> inline Stack<string>& Stacks::getStr<int>(){ return cs; }
    
    /*!
     * @class Trace
     * @brief used for tracing stack outputs for backprop algorithm.
     */
    struct Trace
    {
        vector<ArrayXd> f;
        vector<ArrayXi> c;
        vector<ArrayXb> b;
        
        template <typename T> inline vector<Eigen::Array<T,Eigen::Dynamic,1> >& get()
        {
            return get<Eigen::Array<T,Eigen::Dynamic,1> >();
        }
        
        template <typename T> unsigned int size()
        {
            return get<T>().size();
        }
    };
    
    template <> inline vector<ArrayXd>& Trace::get(){ return f; }
        
    template <> inline vector<ArrayXb>& Trace::get(){ return b; }
    
    template <> inline vector<ArrayXi>& Trace::get(){ return c; }
}

#endif
