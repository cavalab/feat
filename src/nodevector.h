/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEVECTOR_H
#define NODEVECTOR_H

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class NodeVector
     * @brief an extension of a vector of unique pointers to nodes 
     */
    struct NodeVector : public std::vector<std::unique_ptr<Node>> {
        
        NodeVector() = default;
        ~NodeVector() = default; 
        NodeVector(const NodeVector& other)
        {
            /* std::cout<<"in NodeVector(const NodeVector& other)\n"; */
            this->resize(0);
            for (const auto& p : other)
                this->push_back(p->clone());
        }
        NodeVector(NodeVector && other) = default;
        /* { */
        /*     std::cout<<"in NodeVector(NodeVector&& other)\n"; */
        /*     for (const auto& p : other) */
        /*         this->push_back(p->clone()); */
        /* } */
        NodeVector& operator=(NodeVector const& other)
        { 

            /* std::cout << "in NodeVector& operator=(NodeVector const& other)\n"; */
            this->resize(0);
            for (const auto& p : other)
                this->push_back(p->clone());
            return *this; 
        }        
        NodeVector& operator=(NodeVector && other) = default;
         

    }; //NodeVector
} // FT
#endif
