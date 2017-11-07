/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_H
#define NODE_H

#include<map>
using std::vector;
using std::string;
using std::map;
using Eigen::ArrayXd;
#define MAX_DBL std::numeric_limits<double>::max()
#define MIN_DBL std::numeric_limits<double>::min()



namespace FT{

    //////////////////////////////////////////////////////////////////////////////// Declarations
   double NEAR_ZERO = 0.0000001;
    /*!
     * @class Node
     * @brief Represents nodes in a program.
     */
    class Node
    {       
        public:
            string name;              				///< node type
            char otype;             				///< output type
            std::map<char, unsigned int> arity;		///< floating arity of the operator 
            int complexity;         ///< complexity of node
            
            virtual ~Node(){}
           
            /// Evaluates the node and updates the stack states. 
            virtual void evaluate(const MatrixXd& X, const VectorXd& y,vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b) = 0; 

            /// evaluates the node symbolically
            virtual void eval_eqn(vector<string>& stack_f, vector<string>& stack_b) = 0;

            // total arity
            unsigned int total_arity(){ return arity['f'] + arity['b']; };

            /// limits node output to be between MIN_DBL and MAX_DBL
            ArrayXd limited(ArrayXd x)
            {
                
                x = (Eigen::isinf(x)).select(MAX_DBL,x);
                x = (Eigen::isnan(x)).select(0,x);
                //x = (x < MIN_DBL).select(MIN_DBL,x);
                return x;
            };

            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity(map<char, vector<unsigned int>>& cstack)
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
                        c_args += cstack[a.first].back();
                        cstack[a.first].pop_back();

                    }
                }
                cstack[otype].push_back(complexity*c_args);
               
            }
            
            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity_db(map<char, vector<string>>& cstack)
            {
                /*! Complexity of a node \f$ n \f$ with \f$ k \f$ arguments is defined as 
                 *  
                 *  \f$ C(n) = c_n * (\sum_{a=1}^k C(a)) \f$
                 *
                 *  The complexity of a program is the complexity of its root/head node. 
                 */              
                string c_args="1";                         // sum complexity of the arguments 
                if (total_arity() ==0)
                    cstack[otype].push_back(c_args);
                else{
                    for (const auto& a: arity)
                    {
                        for (unsigned int i = 0; i< a.second; ++i)
                        {
                            c_args = "(" + c_args + "+" + cstack[a.first].back() + ")";
                            cstack[a.first].pop_back();

                        }
                    }
                    cstack[otype].push_back(std::to_string(complexity) + "*" + c_args);
                }
            }
    };
}
#endif
