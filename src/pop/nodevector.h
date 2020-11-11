/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEVECTOR_H
#define NODEVECTOR_H
#include <memory>
#include "nodewrapper.h"
#include "nodemap.h"
#include "../util/error.h"

namespace FT{
    
    namespace Pop{
    
        using namespace Op;
        ////////////////////////////////////////////////////////////////////////////////// Declarations

        /*!
         * @class NodeVector
         * @brief an extension of a vector of unique pointers to nodes 
         */
        struct NodeVector : public std::vector<std::unique_ptr<Node>> {
            
            NodeVector();
            
            ~NodeVector();
            
            NodeVector(const NodeVector& other);
            
            NodeVector(NodeVector && other);
            /* { */
            /*     std::cout<<"in NodeVector(NodeVector&& other)\n"; */
            /*     for (const auto& p : other) */
            /*         this->push_back(p->clone()); */
            /* } */
            
            NodeVector& operator=(NodeVector const& other);
            
            NodeVector& operator=(NodeVector && other);
            
            /// returns vector of raw pointers to nodes in [start,end], or all if both are zero
            vector<Node*> get_data(int start=0,int end=0);

            /// returns indices of root nodes 
            vector<size_t> roots() const;

            size_t subtree(size_t i, char otype='0', string indent="> ") const;
            
            void set_weights(vector<vector<float>>& weights);
            
            vector<vector<float>> get_weights();
            
            bool is_valid_program(unsigned num_features, 
                                  vector<string> longitudinalMap);
       
            void make_tree(const NodeVector& functions, 
                           const NodeVector& terminals, int max_d,  
                           const vector<float>& term_weights, 
                           const vector<float>& op_weights, 
                           char otype, const vector<char>& term_types);

            void make_program(const NodeVector& functions, 
                              const NodeVector& terminals, int max_d, 
                              const vector<float>& term_weights, 
                              const vector<float>& op_weights, 
                              int dim, char otype, 
                              vector<string> longitudinalMap, const vector<char>& term_types);
            
        }; //NodeVector
        // serializatoin
        // forward declaration
        void to_json(json& j, const NodeVector& nv);
        void from_json(const json& j, NodeVector& nv);
        
    }
} // FT
#endif

