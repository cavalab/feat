/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodevector.h"

namespace FT{
    
    NodeVector::NodeVector() = default;
    
    NodeVector::~NodeVector() = default; 

    NodeVector::NodeVector(NodeVector && other) = default;

    NodeVector& NodeVector::operator=(NodeVector && other) = default;

    NodeVector::NodeVector(const NodeVector& other)
    {
        /* std::cout<<"in NodeVector(const NodeVector& other)\n"; */
        this->resize(0);
        for (const auto& p : other)
            this->push_back(p->clone());
    }
    
    /* { */
    /*     std::cout<<"in NodeVector(NodeVector&& other)\n"; */
    /*     for (const auto& p : other) */
    /*         this->push_back(p->clone()); */
    /* } */
    
    NodeVector& NodeVector::operator=(NodeVector const& other)
    { 

        /* std::cout << "in NodeVector& operator=(NodeVector const& other)\n"; */
        this->resize(0);
        for (const auto& p : other)
            this->push_back(p->clone());
        return *this; 
    }
            
    vector<Node*> NodeVector::get_data(int start,int end)
    {
        vector<Node*> v;
        if (end == 0)
        {
            if (start == 0)
                end = this->size();
            else
                end = start;
        }
        for (unsigned i = start; i<=end; ++i)
            v.push_back(this->at(i).get());

        return v;
    }

    /// returns indices of root nodes 
    vector<size_t> NodeVector::roots() const
    {
        // find "root" nodes of program, where roots are final values that output 
        // something directly to the stack
        // assumes a program's subtrees to be contiguous
         
        vector<size_t> indices;     // returned root indices
        int total_arity = -1;       //end node is always a root
        for (size_t i = this->size(); i>0; --i)   // reverse loop thru program
        {    
            if (total_arity <= 0 ){ // root node
                indices.push_back(i-1);
                total_arity=0;
            }
            else
                --total_arity;
           
            total_arity += this->at(i-1)->total_arity(); 
           
        }
       
        return indices; 
    }

    size_t NodeVector::subtree(size_t i, char otype) const 
    {

       /*!
        * finds index of the end of subtree in program with root i.
        
        * Input:
        
        *		i, root index of subtree
        
        * Output:
        
        *		last index in subtree, <= i
        
        * note that this function assumes a subtree's arguments to be contiguous in the program.
        */
       
       size_t tmp = i;
       assert(i>=0 && "attempting to grab subtree with index < 0");
              
       if (this->at(i)->total_arity()==0)    // return this index if it is a terminal
           return i;
       
       std::map<char, unsigned int> arity = this->at(i)->arity;

       if (otype!='0')  // if we are recursing (otype!='0'), we need to find 
                        // where the nodes to recurse are.  
       {
           while (i>0 && this->at(i)->otype != otype) --i;    
           assert(this->at(i)->otype == otype && "invalid subtree arguments");
       }
              
       for (unsigned int j = 0; j<arity['f']; ++j)  
           i = subtree(--i,'f');                   // recurse for floating arguments      
       
       size_t i2 = i;                              // index for second recursion
       for (unsigned int j = 0; j<arity['b']; ++j)
           i2 = subtree(--i2,'b');
       
       size_t i3 = i2;                 // recurse for longitudinal arguments
       for (unsigned int j = 0; j<arity['z']; ++j)
           i3 = subtree(--i3,'z');
       
       size_t i4 = i3;                 // recurse for categorical arguments
       for (unsigned int j = 0; j<arity['c']; ++j)
           i4 = subtree(--i4,'c'); 
       
       return std::min(i,i4);
    }
    
    void NodeVector::set_weights(vector<vector<double>>& weights)
    {
        if (weights.size()==0) return;
        int count = 0;
        for (unsigned i = 0; i< this->size(); ++i)
        {
            if (this->at(i)->isNodeDx())
            {
                NodeDx* nd = dynamic_cast<NodeDx*>(this->at(i).get());
                
                if (weights.at(count).size() == nd->W.size())
                    nd->W = weights.at(count);
                else
                {
                    string error = "mismatch in size btw weights[" + to_string(count) + "] and W\n";
                    error += "weights[" + to_string(count) + "].size() (" + to_string(weights[count].size()) + ") != W.size() ("+ to_string(nd->W.size()) + "\n";
                    HANDLE_ERROR_THROW(error);
                }
                ++count;
            }
        }
    }
    
    vector<vector<double>> NodeVector::get_weights()
    {
        vector<vector<double>> weights;
        for (unsigned i = 0; i< this->size(); ++i)
        {
            if (this->at(i)->isNodeDx())
            {
                weights.push_back(dynamic_cast<NodeDx*>(this->at(i).get())->W); 
            }
        }
        return weights;
    }
} // FT
