/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "node.h"

namespace FT{

    namespace Pop{
        namespace Op{
            Node::Node()
            {
                arity['f'] = 0;
                arity['b'] = 0;
                arity['c'] = 0;
                arity['z'] = 0;
            }
            
            unsigned int Node::total_arity()
            {
                if(arity.find('f') == arity.end())
                    arity['f'] = 0;
                
                if(arity.find('b') == arity.end())
                    arity['b'] = 0;
                    
                if(arity.find('c') == arity.end())
                    arity['c'] = 0;
                
                if(arity.find('z') == arity.end())
                    arity['z'] = 0;
                        
                return arity['f'] + arity['b'] + arity['c'] + arity['z'];
            }

            /// limits node output to be between MIN_FLT and MAX_FLT
            ArrayXf Node::limited(ArrayXf x)
            {
                x = (isnan(x)).select(0,x);
                x = (x < MIN_FLT).select(MIN_FLT,x);
                x = (x > MAX_FLT).select(MAX_FLT,x);
                return x;
            };

            /// evaluates complexity of this node in the context of its child nodes.
            void Node::eval_complexity(map<char, vector<unsigned int>>& cstate)
            {
                /*! Complexity of a node \f$ n \f$ with \f$ k \f$ arguments is defined as 
                 *  
                 *  \f$ C(n) = c_n * (\sum_{a=1}^k C(a)) \f$
                 *
                 *  The complexity of a program is the complexity of its root/head node. 
                 */              
                int c_args=1;                         // sum complexity of the arguments 
                for (const auto& a: arity)
                {
                    for (unsigned int i = 0; i< a.second; ++i)
                    {
                        c_args += cstate[a.first].back();
                        cstate[a.first].pop_back();

                    }
                }
                cstate[otype].push_back(complexity*c_args);
               
            }

            /// evaluates complexity of this node in the context of its child nodes.
            void Node::eval_complexity_db(map<char, vector<string>>& cstate)
            {
                /*! Complexity of a node \f$ n \f$ with \f$ k \f$ arguments is defined as 
                 *  
                 *  \f$ C(n) = c_n * (\sum_{a=1}^k C(a)) \f$
                 *
                 *  The complexity of a program is the complexity of 
                 *  its root/head node. 
                 */              
                // sum complexity of the arguments 

                string c_args="1";                                         
                if (total_arity() ==0)
                    cstate[otype].push_back(c_args);
                else{
                    for (const auto& a: arity)
                    {
                        for (unsigned int i = 0; i< a.second; ++i)
                        {
                            c_args = "(" + c_args + "+" 
                                + cstate[a.first].back() + ")";
                            cstate[a.first].pop_back();

                        }
                    }
                    cstate[otype].push_back(std::to_string(complexity) 
                            + "*" + c_args);
                }
            }

            /// makes a unique copy of this node
            std::unique_ptr<Node> Node::clone() const 
            { return std::unique_ptr<Node>(clone_impl()); }

            /// makes a randomized unique copy ofnode
            std::unique_ptr<Node> Node::rnd_clone() const 
            {return std::unique_ptr<Node>(rnd_clone_impl());}

        }
    }
}
