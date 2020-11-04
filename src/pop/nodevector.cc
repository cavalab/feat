/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "nodevector.h"
using namespace FT::Pop::Op; 
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

size_t NodeVector::subtree(size_t i, char otype, string indent) const 
{

   /*!
    * finds index of the end of subtree in program with root i.
    
    * Input:
    
    *		i, root index of subtree
    
    * Output:
    
    *		last index in subtree, <= i
    
    * note that this function assumes a subtree's arguments to be 
    * contiguous in the program.
    */
   size_t tmp = i;
   if (i<0 || i > this->size())
       THROW_LENGTH_ERROR("Attempting got grab subtree with index " 
               + to_string(i) + " and program size " 
               + to_string(this->size()));
          
   /* cout << indent << "getting subtree(" << i << "," */ 
   /*     << otype << ") for " << this->at(i)->name */ 
   /*     << " of type " << this->at(i)->otype << endl; */
   // return this index if it is a terminal
   if (this->at(i)->total_arity()==0)                   
   {
       return i;
   }
   
   std::map<char, unsigned int> arity = this->at(i)->arity;

   /* cout << indent << "otype: " << otype << endl; */

   if (otype!='0')  // if we are recursing (otype!='0'), we need to find 
                    // where the nodes to recurse are.  
   {
       while (i>0 && this->at(i)->otype != otype) --i;    

       if (this->at(i)->otype != otype)
           THROW_INVALID_ARGUMENT("invalid subtree arguments");
   }
 
   /* cout << indent << "i at 125: " << i << "\n"; */
   // recurse for floating arguments
   for (unsigned int j = 0; j<arity.at('f'); ++j)  
       i = subtree(--i,'f', indent+indent);                         
   /* cout << indent << "i at 129: " << i << "\n"; */
   // recurse for boolean
   size_t i2 = i;                              
   for (unsigned int j = 0; j<arity.at('b'); ++j)
       i2 = subtree(--i2,'b', indent+indent);
   /* cout << indent << "i2 at 134: " << i2 << "\n"; */
   // recurse for categorical arguments
   size_t i3 = i2;                 
   for (unsigned int j = 0; j<arity.at('c'); ++j)
       i3 = subtree(--i3,'c', indent+indent); 
   /* cout << indent << "i3 at 139: " << i3 << "\n"; */
   // recurse for longitudinal arguments
   size_t i4 = i3;                 
   for (unsigned int j = 0; j<arity.at('z'); ++j)
       i4 = subtree(--i4,'z', indent+indent);
   /* cout << indent << "i4 at 145: " << i4 << "\n"; */
   /* cout << indent << "returning min(" << i << "," << i4 << ")\n"; */ 

   return std::min(i,i4);
}

void NodeVector::set_weights(vector<vector<float>>& weights)
{
    if (weights.size()==0) return;
    int count = 0;
    /* int dx_node_count = 0; */
    /* for (unsigned i = 0; i< this->size(); ++i) */
    /* { */
    /*     if (this->at(i)->isNodeDx()) */
    /*     { */
    /*         ++dx_node_count; */
    /*     } */
    /* } */
    /* if (weights.size() != dx_node_count) */
    /* {} */
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
                         "].size() (" + to_string(weights.at(count).size()) +
                         ") != W.size() ("+ to_string(nd->W.size()) + "\n";
                         
                THROW_LENGTH_ERROR(error);
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
        // learning nodes are set for fit or predict mode
        if (n->isNodeTrain())                     
            dynamic_cast<NodeTrain*>(n.get())->train = false;
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
            if (terminals.at(i)->otype == otype) // grab terminals matching output type
            {
                ti.push_back(i);
                tw.push_back(term_weights.at(i));                    
            }
                
        }
        /* cout << "valid terminals: "; */
        /* for (unsigned i = 0; i < ti.size(); ++i) */ 
        /*     cout << terminals.at(ti.at(i))->name << "(" << terminals.at(ti.at(i))->otype << ", " */ 
        /*          << tw.at(i) << "), "; */ 
        /* cout << "\n"; */
        
        if(ti.size() > 0 && tw.size() > 0)
        {
            auto t = terminals.at(r.random_choice(ti,tw))->clone();
            /* std::cout << "chose " << t->name << " "; */
            push_back(t->rnd_clone());
        }
        else
        {
            string ttypes = "";
            for (const auto& t : terminals)
                ttypes += t->name + ": " + t->otype + "\n";
            THROW_RUNTIME_ERROR("Error: make_tree couldn't find properly typed terminals\n"
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
        bool zterms = in(term_types, 'z');   // are there long terminals?
        /* std::cout << "fterms: " << fterms << ", bterms: " << bterms */ 
        /*     << ",cterms: " << cterms */ 
        /*   << ",zterms: " << zterms << "\n"; */ 
        for (size_t i = 0; i<functions.size(); ++i)
            if (functions.at(i)->otype==otype &&
                (max_d>1 || functions.at(i)->arity.at('f')==0 || fterms) &&
                (max_d>1 || functions.at(i)->arity.at('b')==0 || bterms) &&
                (max_d>1 || functions.at(i)->arity.at('c')==0 || cterms) &&
                (max_d>1 || functions.at(i)->arity.at('z')==0 || zterms))
            {
                fi.push_back(i);
                fw.push_back(op_weights.at(i));
            }
        
        // if there are no valid functions, add a terminal
        if (fi.size()==0)
        {
            make_tree(functions, terminals, 0, term_weights, 
                        op_weights, otype, term_types);
            return;

        }
        if (fi.size() == 0)
            THROW_RUNTIME_ERROR("The operator set specified "
                    "results in incomplete programs.");
        
        // append a random choice from fs            
        /* cout << "function choices: \n"; */
        /* for (unsigned fis =0; fis < fi.size(); ++fis) */
        /*     cout << "(" << functions[fi[fis]]->name << "," << fw[fis] << ") ,"; */
        /* cout << "\n"; */
        
        push_back(functions.at(r.random_choice(fi,fw))->rnd_clone());
        
        /* std::unique_ptr<Node> chosen(back()->clone()); */
        map<char, unsigned> chosen_arity = back()->arity;
        // recurse to fulfill the arity of the chosen function
        vector<char> type_order = {'f','b','c','z'};
        for (auto type : type_order)
        {
            for (size_t i = 0; i < chosen_arity.at(type); ++i)
            {
                make_tree(functions, terminals, max_d-1, term_weights, 
                        op_weights, type, term_types);
            }

        }
    }
    
    /* std::cout << "finished program: ["; */
    /* for (const auto& p : *(this) ) std::cout << p->name << " "; */
}

void NodeVector::make_program(const NodeVector& functions, 
                              const NodeVector& terminals, int max_d, 
                              const vector<float>& term_weights, 
                              const vector<float>& op_weights, 
                              int dim, char otype, 
                              vector<string> longitudinalMap, 
                              const vector<char>& term_types)
{
    for (unsigned i = 0; i<dim; ++i)    // build trees
        make_tree(functions, terminals, max_d, term_weights, op_weights, otype, 
                  term_types);
    
    // reverse program so that it is post-fix notation
    std::reverse(begin(), end());
    assert(is_valid_program(terminals.size(), longitudinalMap));
}

//serialization
void to_json(json& j, const NodeVector& nv)
{
    /* vector<json> program; */
    for (const auto& n : nv)
    {
        json k;
        // cast different types of nodes
        if (typeid(*n) == typeid(NodeSplit<float>))
            Op::to_json(k, *dynamic_cast<NodeSplit<float>*>(n.get()));
        
        else if (typeid(*n) == typeid(NodeSplit<int>))
            Op::to_json(k, *dynamic_cast<NodeSplit<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzySplit<float>))
            Op::to_json(k, *dynamic_cast<NodeFuzzySplit<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzySplit<int>))
            Op::to_json(k, *dynamic_cast<NodeFuzzySplit<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzyFixedSplit<float>))
            Op::to_json(k, *dynamic_cast<NodeFuzzyFixedSplit<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzyFixedSplit<int>))
            Op::to_json(k, *dynamic_cast<NodeFuzzyFixedSplit<int>*>(n.get()));

        else if (n->isNodeTrain())                     
            Op::to_json(k, *dynamic_cast<NodeTrain*>(n.get()));

        else if (n->isNodeDx())                     
            Op::to_json(k, *dynamic_cast<NodeDx*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<float>))
            Op::to_json(k, *dynamic_cast<NodeVariable<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<int>))
            Op::to_json(k, *dynamic_cast<NodeVariable<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<bool>))
            Op::to_json(k, *dynamic_cast<NodeVariable<bool>*>(n.get()));

        else if (typeid(*n) == typeid(NodeConstant))
            Op::to_json(k, *dynamic_cast<NodeConstant*>(n.get()));

        else
            Op::to_json(k, *n);

        j.push_back(k);
    }
    
}
void from_json(const json& j, NodeVector& nv)
{
    for (const auto& k : j)
    {
        string node_name = k.at("name").get<string>();

        if (Op::NM.node_map.find(node_name) == Op::NM.node_map.end())
        {
            node_name = k.at("name").get<string>() + "_" 
                               + to_string(k.at("otype").get<char>());
            if (Op::NM.node_map.find(node_name) == Op::NM.node_map.end())
                THROW_INVALID_ARGUMENT(node_name + " not found");
        }

        auto n = NM.node_map[node_name]->clone();
        // cast different types of nodes
        if (typeid(*n) == typeid(NodeSplit<float>))
            Op::from_json(k, *dynamic_cast<NodeSplit<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeSplit<int>))
            Op::from_json(k, *dynamic_cast<NodeSplit<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzySplit<int>))
            Op::from_json(k, *dynamic_cast<NodeFuzzySplit<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzySplit<float>))
            Op::from_json(k, *dynamic_cast<NodeFuzzySplit<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzyFixedSplit<float>))
            Op::from_json(k, *dynamic_cast<NodeFuzzyFixedSplit<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeFuzzyFixedSplit<int>))
            Op::from_json(k, *dynamic_cast<NodeFuzzyFixedSplit<int>*>(n.get()));

        else if (n->isNodeTrain())                     
            Op::from_json(k, *dynamic_cast<NodeTrain*>(n.get()));

        else if (n->isNodeDx())                     
            Op::from_json(k, *dynamic_cast<NodeDx*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<float>))
            Op::from_json(k, *dynamic_cast<NodeVariable<float>*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<int>))
            Op::from_json(k, *dynamic_cast<NodeVariable<int>*>(n.get()));

        else if (typeid(*n) == typeid(NodeVariable<bool>))
            Op::from_json(k, *dynamic_cast<NodeVariable<bool>*>(n.get()));

        else if (typeid(*n) == typeid(NodeConstant))
            Op::from_json(k, *dynamic_cast<NodeConstant*>(n.get()));

        else
            Op::from_json(k, *n);

        // check
        json k2;
        nv.push_back(n->clone());
        /* json k2; */
        /* Op::to_json(k2, *nv.back()); */
        /* cout << "after from_json call: " << k2.dump() << endl; */

    }
    json check;
    to_json(check, nv);
}

} // Pop
} // FT
