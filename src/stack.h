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

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
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
            Stack();
            
            ///< population pushes element at back of vector
            void push(type element);
            
            ///< pops element from back of vector and removes it
            type pop();
            
            ///< returns true or false depending on stack is empty or not
            bool empty();
            
            ///< returns size of stack
            unsigned int size();
            
            ///< returns top element of stack
            type& top();
            
            ///< returns element at particular location in stack
            type& at(int i);
            
            ///< clears the stack
            void clear();
            
            ///< returns start iterator of stack
            typename vector<type>::iterator begin();
            
            ///< returns end iterator of stack
            typename vector<type>::iterator end();
            
            ///< returns const start iterator of stack
            typename vector<type>::const_iterator begin() const;
            
            ///< returns const iterator of stack
            typename vector<type>::const_iterator end() const;
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
        bool check(std::map<char, unsigned int> &arity);
        
        ///< checks if arity of node provided satisfies the node names in various string stacks
        bool check_s(std::map<char, unsigned int> &arity);
    };
}

#endif
