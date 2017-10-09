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
using Eigen::ArrayXi;


namespace FT{

    //////////////////////////////////////////////////////////////////////////////// Declarations
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
            
            Node()
            {
            	//std::cerr << "error in parent node.h : invalid constructor called";
				//throw;
            }
            /*!
             * @brief Evaluates the node and updates the stack states. 
             */            
            virtual void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXi>& stack_b) = 0;

            /*!
             * @brief evaluates the node symbolically
             */
            virtual void eval_eqn(vector<string>& stack_f, vector<string>& stack_b) = 0;

            // total arity
            unsigned int total_arity(){ return arity['f'] + arity['b']; };
    };
}
#endif
