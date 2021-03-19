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
#include "../../init.h"
#include "../../dat/state.h"
#include "../../dat/data.h"
#include "../../util/rnd.h"
#include "../../util/error.h"
#include "../../util/utils.h"
using std::vector;
using std::string;
using std::map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
#define MAX_FLT std::numeric_limits<float>::max()
#define MIN_FLT std::numeric_limits<float>::lowest()

#define MAX_INT std::numeric_limits<int>::max()
#define MIN_INT std::numeric_limits<int>::lowest()

#ifdef USE_CUDA
    #include "../cuda-op/kernels.h"
#endif


namespace FT{

    using namespace Util;
    using namespace Dat;

namespace Pop{
/**
 * @namespace FT::Pop::Op
 * @brief namespace representing various operations on population individuals used in Feat
 */
namespace Op{
    //////////////////////////////////////////////////////////////////////////////// Declarations
     /*!
     * @class Node
     * @brief Represents nodes in a program.
     */
    class Node
    {       
        public:
            string name;              				///< node type
            string variable_name;              		///< variable name, if any
            char otype;             				///< output type
            std::map<char, unsigned int> arity;		///< floating arity of the operator 
            int complexity;         ///< complexity of node
            int visits = 0;
            
            Node();

            virtual ~Node(){}
           
            /// Evaluates the node and updates the state states. 
            virtual void evaluate(const Data& data, State& state) = 0; 

            /// evaluates the node symbolically
            virtual void eval_eqn(State& state) = 0;

            // total arity
            unsigned int total_arity();

            /// limits node output to be between MIN_FLT and MAX_FLT
            ArrayXf limited(ArrayXf x);

            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity(map<char, vector<unsigned int>>& cstate);
            
            /// evaluates complexity of this node in the context of its child nodes.
            void eval_complexity_db(map<char, vector<string>>& cstate);

            /// check of node type
            virtual bool isNodeDx() {return false;};
            virtual bool isNodeTrain() {return false;};

            /// makes a unique copy of this node
            std::unique_ptr<Node> clone() const;
            
            /// makes a randomized unique copy ofnode
            std::unique_ptr<Node> rnd_clone() const;
        
        protected:
            virtual Node* clone_impl() const = 0;
            virtual Node* rnd_clone_impl() const = 0;
    };
   
    // macro to define from_json and to_json
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Node, name, otype, arity, complexity, visits)
}
}
}
#endif
