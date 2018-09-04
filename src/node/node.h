/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_H
#define NODE_H

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "../init.h"
#include "../stack.h"
#include "../data.h"
#include "../rnd.h"
using std::vector;
using std::string;
using std::map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
#define MAX_DBL std::numeric_limits<double>::max()
#define MIN_DBL std::numeric_limits<double>::lowest()
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
            int visits = 0;
            
            Node();

            virtual ~Node(){}
           
            /// Evaluates the node and updates the stack states. 
            virtual void evaluate(const Data& data, Stacks& stack) = 0; 

            /// evaluates the node symbolically
            virtual void eval_eqn(Stacks& stack) = 0;

            // total arity
            unsigned int total_arity();

            /// limits node output to be between MIN_DBL and MAX_DBL
            ArrayXd limited(ArrayXd x);

            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity(map<char, vector<unsigned int>>& cstack);
            
            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity_db(map<char, vector<string>>& cstack);

            /// check of node type
            virtual bool isNodeDx() {return false;};

            /// makes a unique copy of this node
            std::unique_ptr<Node> clone()const;
            
            /// makes a randomized unique copy ofnode
            std::unique_ptr<Node> rnd_clone() const;
        
        protected:
            virtual Node* clone_impl() const = 0;
            virtual Node* rnd_clone_impl() const = 0;
    };
}
#endif
