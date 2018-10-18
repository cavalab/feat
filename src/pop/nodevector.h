/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEVECTOR_H
#define NODEVECTOR_H
#include <memory>
#include "node/node.h"
#include "node/n_Dx.h"

#include "error.h"
namespace FT{
    
    namespace Pop{
    
        using namespace NodeSpace;
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

            size_t subtree(size_t i, char otype='0') const;
            
            void set_weights(vector<vector<double>>& weights);
            
            vector<vector<double>> get_weights();
            
        }; //NodeVector
        
    }
} // FT
#endif
