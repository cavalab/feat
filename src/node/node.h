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
            /*!
             * @brief Evaluates the node and updates the stack states. 
             */            
            virtual void evaluate(const MatrixXd& X, const VectorXd& y,vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b) = 0; 

            /*!
             * @brief evaluates the node symbolically
             */
            virtual void eval_eqn(vector<string>& stack_f, vector<string>& stack_b) = 0;

            // total arity
            unsigned int total_arity(){ return arity['f'] + arity['b']; };

            /// limits node output to be between MIN_DBL and MAX_DBL
            ArrayXd limited(ArrayXd x)
            {
                x = (x > MAX_DBL).select(MAX_DBL,x);
                x = (x < MIN_DBL).select(MIN_DBL,x);
                return x;
            };

            void eval_complexity(map<char, vector<unsigned int>>& cstack)
            {
                

            }
    };
}
#endif
