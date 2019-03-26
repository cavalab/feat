/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodevector.h"

namespace FT{

    namespace Pop{    
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
            // something directly to the state
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
           
            std::reverse(indices.begin(), indices.end()); 
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
        
        void NodeVector::set_weights(vector<vector<float>>& weights)
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
                        string error = "mismatch in size btw weights[" +
                                        to_string(count) +
                                        "] and W\n";
                                        
                        error += "weights[" + to_string(count) + 
                                 "].size() (" + to_string(weights[count].size()) +
                                 ") != W.size() ("+ to_string(nd->W.size()) + "\n";
                                 
                        HANDLE_ERROR_THROW(error);
                    }
                    ++count;
                }
            }
        }
        
        vector<vector<float>> NodeVector::get_weights()
        {
            vector<vector<float>> weights;
            for (unsigned i = 0; i< this->size(); ++i)
            {
                if (this->at(i)->isNodeDx())
                {
                    weights.push_back(dynamic_cast<NodeDx*>(this->at(i).get())->W); 
                }
            }
            return weights;
        }
        
        bool NodeVector::is_valid_program(unsigned num_features, 
                                          vector<string> longitudinalMap)
        {
            /*! checks whether program fulfills all its arities. */
            State state;
            
            std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>> Z;
            
            MatrixXf X = MatrixXf::Zero(num_features,2); 
            VectorXf y = VectorXf::Zero(2);
            
             for(auto key : longitudinalMap)
             {
                Z[key].first.push_back(ArrayXf::Zero(2));
                Z[key].first.push_back(ArrayXf::Zero(2));
                Z[key].second.push_back(ArrayXf::Zero(2));
                Z[key].second.push_back(ArrayXf::Zero(2));
             }
             
            Data data(X, y, Z, false);
            
            unsigned i = 0; 
            for (const auto& n : *this){
                if (state.check(n->arity))
                    n->evaluate(data, state);
                else
                {
                    std::cout << "Error: ";
                    for (const auto& p: *this) std::cout << p->name << " ";
                    std::cout << "is not a valid program because ";
                    std::cout << n->name << " at pos " << i << "is not satisfied\n";
                    return false; 
                }
                ++i;
            }
            return true;
        }
       
        void NodeVector::make_tree(const NodeVector& functions, 
                                   const NodeVector& terminals, int max_d,  
                                   const vector<float>& term_weights,
                                   const vector<float>& op_weights,
                                   char otype, const vector<char>& term_types)
        {  
                    
            /*!
             * recursively builds a program with complete arguments.
             */
            // debugging output
            /* std::cout << "current program: ["; */
            /* for (const auto& p : *(this) ) std::cout << p->name << " "; */
            /* std::cout << "]\n"; */
            /* std::cout << "otype: " << otype << "\n"; */
            /* std::cout << "max_d: " << max_d << "\n"; */

            if (max_d == 0 || r.rnd_flt() < terminals.size()/(terminals.size()+functions.size())) 
            {
                // append terminal 
                vector<size_t> ti;  // indices of valid terminals 
                vector<float> tw;  // weights of valid terminals
                /* cout << "terminals: " ; */
                /* for (const auto& t : terminals) cout << t->name << "(" << t->otype << "),"; */ 
                /* cout << "\n"; */
                
                for (size_t i = 0; i<terminals.size(); ++i)
                {
                    if (terminals[i]->otype == otype) // grab terminals matching output type
                    {
                        ti.push_back(i);
                        tw.push_back(term_weights[i]);                    
                    }
                        
                }
                /* cout << "valid terminals: "; */
                /* for (unsigned i = 0; i < ti.size(); ++i) */ 
                /*     cout << terminals[ti[i]]->name << "(" << terminals[ti[i]]->otype << ", " */ 
                /*          << tw[i] << "), "; */ 
                /* cout << "\n"; */
                
                if(ti.size() > 0 && tw.size() > 0)
                {
                    auto t = terminals[r.random_choice(ti,tw)]->clone();
                    /* std::cout << "chose " << t->name << " "; */
                    push_back(t->rnd_clone());
                }
                else
                {
                    string ttypes = "";
                    for (const auto& t : terminals)
                        ttypes += t->name + ": " + t->otype + "\n";
                    HANDLE_ERROR_THROW("Error: make_tree couldn't find properly typed terminals\n"
                                       + ttypes);
                }
            }
            else
            {
                // let fi be indices of functions whose output type matches otype and, if max_d==1,
                // with no boolean inputs (assuming all input data is floating point) 
                vector<size_t> fi;
                vector<float> fw;  // function weights
                bool fterms = in(term_types, 'f');   // are there floating terminals?
                bool bterms = in(term_types, 'b');   // are there boolean terminals?
                bool cterms = in(term_types, 'c');   // are there categorical terminals?
                bool zterms = in(term_types, 'z');   // are there boolean terminals?
                /* std::cout << "bterms: " << bterms << ",cterms: " << cterms 
                 * << ",zterms: " << zterms << "\n"; */
                for (size_t i = 0; i<functions.size(); ++i)
                    if (functions[i]->otype==otype &&
                        (max_d>1 || functions[i]->arity['f']==0 || fterms) &&
                        (max_d>1 || functions[i]->arity['b']==0 || bterms) &&
                        (max_d>1 || functions[i]->arity['c']==0 || cterms) &&
                        (max_d>1 || functions[i]->arity['z']==0 || zterms))
                    {
                        fi.push_back(i);
                        fw.push_back(op_weights.at(i));
                    }
                
                if (fi.size()==0){

                    if(otype == 'z')
                    {
                        make_tree(functions, terminals, 0, term_weights, op_weights, 'z', 
                                  term_types);
                        return;
                    }
                    else if (otype == 'c')
                    {
                        make_tree(functions, terminals, 0, term_weights, op_weights, 'c', 
                                  term_types);
                        return;
                    }
                    else{            
                        std::cout << "---\n";
                        std::cout << "f1.size()=0. current program: ";
                        for (const auto& p : *(this)) std::cout << p->name << " ";
                        std::cout << "\n";
                        std::cout << "otype: " << otype << "\n";
                        std::cout << "max_d: " << max_d << "\n";
                        std::cout << "functions: ";
                        for (const auto& f: functions) std::cout << f->name << " ";
                        std::cout << "\n";
                        std::cout << "---\n";
                    }
                }
                
                assert(fi.size() > 0 && "The operator set specified results in incomplete programs.");
                
                // append a random choice from fs            
                /* cout << "function choices: \n"; */
                /* for (unsigned fis =0; fis < fi.size(); ++fis) */
                /*     cout << "(" << functions[fi[fis]]->name << "," << fw[fis] << ") ,"; */
                /* cout << "\n"; */
                
                push_back(functions[r.random_choice(fi,fw)]->rnd_clone());
                
                /* std::cout << "back(): " << back()->name << "\n"; */ 
                std::unique_ptr<Node> chosen(back()->clone());
                /* std::cout << "chosen: " << chosen->name << "\n"; */ 
                /* std::cout << "continue?" ; */
                /* int cont = 0; */
                /* std::cin >>  cont ; */ 
                /* if (cont != 1) */
                /*     HANDLE_ERROR_THROW("exiting"); */
                // recurse to fulfill the arity of the chosen function
                for (size_t i = 0; i < chosen->arity['f']; ++i)
                    make_tree(functions, terminals, max_d-1, term_weights, op_weights, 'f', 
                              term_types);
                for (size_t i = 0; i < chosen->arity['b']; ++i)
                    make_tree(functions, terminals, max_d-1, term_weights, op_weights, 'b', 
                              term_types);
                for (size_t i = 0; i < chosen->arity['c']; ++i)
                    make_tree(functions, terminals, max_d-1, term_weights, op_weights, 'c', 
                              term_types);
                for (size_t i = 0; i < chosen->arity['z']; ++i)
                    make_tree(functions, terminals, max_d-1, term_weights, op_weights, 'z', 
                              term_types);
            }
            
            /* std::cout << "finished program: ["; */
            /* for (const auto& p : *(this) ) std::cout << p->name << " "; */
        }

        void NodeVector::make_program(const NodeVector& functions, 
                                      const NodeVector& terminals, int max_d, 
                                      const vector<float>& term_weights, 
                                      const vector<float>& op_weights, 
                                      int dim, char otype, 
                                      vector<string> longitudinalMap, const vector<char>& term_types)
        {
            for (unsigned i = 0; i<dim; ++i)    // build trees
                make_tree(functions, terminals, max_d, term_weights, op_weights, otype, term_types);
            
            // reverse program so that it is post-fix notation
            std::reverse(begin(), end());
            assert(is_valid_program(terminals.size(), longitudinalMap));
        }
        
    }
} // FT
