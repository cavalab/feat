/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "params.h"
#include "util/utils.h"

namespace FT{

using namespace Util;
using namespace Pop::Op;
    
Parameters::Parameters(int pop_size, int gens, string ml, bool classification, 
        int max_stall, char ot, int verbosity, string fs, float cr, 
        float root_xor, unsigned int max_depth, 
        unsigned int max_dim, bool constant, string obj, bool sh, float sp, 
        float fb, string sc, string fn, bool bckprp, int iters, float lr,
        int bs, bool hclimb, int maxt, bool res_xo, bool stg_xo, 
        bool stg_xo_tol, bool sftmx, bool nrm, bool corr_mut, bool tune_init,
        bool tune_fin):    
        pop_size(pop_size),
        gens(gens),
        ml(ml),
        classification(classification),
        max_stall(max_stall), 
        cross_rate(cr),
        root_xo_rate(root_xor),
        max_depth(max_depth),
        max_dim(max_dim),
        erc(constant),
        shuffle(sh),
        split(sp),
        otype(ot),
        feedback(fb),
        backprop(bckprp),
        bp(iters, lr, bs),
        hillclimb(hclimb),
        hc(iters, lr),
        max_time(maxt),
        residual_xo(res_xo),
        stagewise_xo(stg_xo),
        stagewise_xo_tol(stg_xo),
        corr_delete_mutate(corr_mut),
        softmax_norm(sftmx),
        normalize(nrm),
        tune_initial(tune_init),
        tune_final(tune_fin),
        scorer(sc)
    {
        set_verbosity(verbosity);
            
        set_functions(fs);
        set_objectives(obj);
        set_feature_names(fn);
        updateSize();     
        set_otypes();
        n_classes = 2;
        set_scorer(sc);
        use_batch = bs>0;
        set_current_gen(0);
    }

Parameters::~Parameters()
{
    
}



/*! checks initial parameter settings before training.
 *  make sure ml choice is valid for problem type.
 *  make sure scorer is set. 
 *  for classification, check clases and find number.
 */
void Parameters::init(const MatrixXf& X, const VectorXf& y)
{
    if (ml == "LinearRidgeRegression" && classification)
    {
        logger.log("Setting ML type to LR",2);
        ml = "LR";            
    }
    if (this->classification)  // setup classification endpoint
    {
       this->set_classes(y);       
    } 
    
    if (this->dtypes.size()==0)    // set feature types if not set
        this->dtypes = find_dtypes(X);
    if (this->verbosity >= 2)
    {
        cout << "X data types: ";
        for (auto dt : this->dtypes)
        {
           cout << dt << ", "; 
        }
        cout << "\n";
    }
    this->set_scorer("", true);
}

/// sets current generation
void Parameters::set_current_gen(int g) { current_gen = g; }

/// sets scorer type
void Parameters::set_scorer(string sc, bool initialized)
{
    string tmp = this->scorer_;
    if (sc.empty()) 
    {
        if (this->scorer.empty() && initialized)
        {
            if (classification && n_classes == 2)
            {
                if (ml.compare("LR") || ml.compare("SVM"))
                    scorer_ = "log";
                else
                    scorer_ = "zero_one";
            }
            else if (classification){
                if (ml.compare("LR") || ml.compare("SVM"))
                    scorer_ = "multi_log";
                else
                    scorer_ = "bal_zero_one";
            }
            else
                scorer_ = "mse";
        }
    }
    else
        scorer_ = sc;

    if (tmp != this->scorer_)
        logger.log("scorer changed to " + scorer_,2);
}

/// sets weights for terminals. 
void Parameters::set_term_weights(const vector<float>& w)
{           
    float u = 1.0/float(terminals.size());
    term_weights.clear();
    if (w.empty())  // set all weights uniformly
    {
        for (unsigned i = 0; i < terminals.size(); ++i)
            term_weights.push_back(u);
    }
    else
    {
        // take abs value of weights
        vector<float> aw = w;
        float weighted_proportion = float(w.size())/float(terminals.size());
        float sum = 0;
        for (unsigned i = 0; i < aw.size(); ++i)
        { 
            aw[i] = fabs(aw[i]); 
            sum += aw[i];
        }
        // normalize weights to one
        for (unsigned i = 0; i < aw.size(); ++i)
        { 
            aw[i] = aw[i]/sum*weighted_proportion;  // awesome!
        }
        int x = 0;
        if (aw.size() != terminals.size())
            THROW_LENGTH_ERROR("There are " + to_string(aw.size()) +
                    "weights and " + to_string(terminals.size()) + 
                    "terminals");
        // assign transformed weights as terminal weights
        for (unsigned i = 0; i < terminals.size(); ++i)
        {
            /* if(terminals[i]->otype == 'z') */
            /*     term_weights.push_back(u); */
            /* else */
            /* { */
                term_weights.push_back((1-feedback)*u + feedback*aw[x]);
                ++x;
            /* } */
        }
           
    }
    string weights = "terminal weights: ";
    for (unsigned i = 0; i < terminals.size(); ++i)
    {
        weights += ("[" 
                + terminals.at(i)->variable_name 
                + " (" + 
                terminals.at(i)->otype + "): " +
                std::to_string(term_weights.at(i)) 
                + "], "); 
    }
    weights += "\n";
    logger.log(weights, 2);
}

void Parameters::updateSize()
{
    max_size = (pow(2,max_depth+1)-1)*max_dim;
}

/// set max depth of programs
void Parameters::set_max_depth(unsigned int max_depth)
{
    this->max_depth = max_depth;
    updateSize();
}

/// set maximum dimensionality of programs
void Parameters::set_max_dim(unsigned int max_dim)
{
    this->max_dim = max_dim;
    updateSize();
}

void Parameters::set_otype(char ot){ otype = ot; set_otypes();}

void Parameters::set_ttypes()
{
    ttypes.clear();
    // set terminal types
    for (const auto& t: terminals)
    {
        if (!in(ttypes,t->otype)) 
            ttypes.push_back(t->otype);
    }
}

/// set the output types of programs
void Parameters::set_otypes(bool terminals_set)
{
    otypes.clear();
    // set output types
    switch (otype)
    { 
        case 'b': otypes.push_back('b'); break;
        case 'f': otypes.push_back('f'); break;
        //case 'c': otypes.push_back('c'); break;
        default: 
        {
            // if terminals are all boolean, remove floating point functions
            if (ttypes.size()==1 && ttypes.at(0)=='b')
            {
                logger.log(string("otypes is size 1 and otypes[0]==b\n") 
                        + string("setting otypes to boolean...\n"),
                        2);
                /* size_t n = functions.size(); */
                /* for (vector<int>::size_type i =n-1; */ 
                /*      i != (std::vector<int>::size_type) -1; i--){ */
                /*     if (functions.at(i)->arity['f'] >0){ */
                /*         logger.log("erasing function " + functions.at(i)->name + "\n", 2); */
                /*         functions.erase(functions.begin()+i); */
                /*     } */
                /* } */
                
                otype = 'b';
                otypes.push_back('b');
            }           
            else
            {
                otypes.push_back('b');
                otypes.push_back('f');
            }
            
            //erasing categorical nodes if no categorical stack exists  
            /* if (terminals_set && !in(ttypes, 'c')) */
            /* { */
            /*     size_t n = functions.size(); */
            /*     for (vector<int>::size_type i =n-1; */ 
            /*          i != (std::vector<int>::size_type) -1; i--){ */
            /*         if (functions.at(i)->arity['c'] >0){ */
            /*             logger.log("erasing function " + functions.at(i)->name + "\n", 2); */
            /*             functions.erase(functions.begin()+i); */
            /*         } */
            /*     } */
            /* } */         
            break;
        }
    }
}

std::unique_ptr<Node> Parameters::createNode(string str,
                                             float d_val,
                                             bool b_val,
                                             size_t loc,
                                             string name)
{
    // variables and constants
    if (str == "x")
    { 
        if(dtypes.size() == 0)
        {
            if (feature_names.size() == 0)
                return std::unique_ptr<Node>(new NodeVariable<float>(loc));
            else
                return std::unique_ptr<Node>(
                        new NodeVariable<float>(loc,'f', 
                            feature_names.at(loc)));
        }
        else if (feature_names.size() == 0)
        {
            switch(dtypes[loc])
            {
                case 'b': 
                    return std::unique_ptr<Node>(
                            new NodeVariable<bool>(loc, dtypes[loc]));
                case 'c': 
                    return std::unique_ptr<Node>(
                            new NodeVariable<int>(loc, dtypes[loc]));
                case 'f': 
                    return std::unique_ptr<Node>(
                                  new NodeVariable<float>(loc, 
                                      dtypes[loc]));
            }
        }
        else
        {
            switch(dtypes[loc])
            {
                case 'b': 
                    return std::unique_ptr<Node>(
                            new NodeVariable<bool>(loc, dtypes[loc],
                                feature_names.at(loc)));
                case 'c': 
                    return std::unique_ptr<Node>(
                            new NodeVariable<int>(loc, dtypes[loc],
                                feature_names.at(loc)));
                
                case 'f': 
                    return std::unique_ptr<Node>(
                            new NodeVariable<float>(loc, dtypes[loc],
                                feature_names.at(loc)));
            }
        }
    }
    else if (str == "z")
        return std::unique_ptr<Node>(new NodeLongitudinal(name));
    else if (NM.node_map.find(str) != NM.node_map.end())
        return NM.node_map[str]->clone();
    else
    {

        cout << "NM.node_map = \n";
        for (auto it = NM.node_map.cbegin(); it != NM.node_map.cend(); )
        {
            cout << it->first << it->second->name << endl;
        }
        THROW_INVALID_ARGUMENT("Error: no node named '" + str + "' exists."); 
    }
    
    return std::unique_ptr<Node>();
}

void Parameters::set_protected_groups(string pg)
{
    if (pg.empty())
        protected_groups.clear();
    else
    {
        pg += ',';      // add delimiter to end
        string delim=",";
        size_t pos = 0;
        string token;
        while ((pos = pg.find(delim)) != string::npos) 
        {
            token = pg.substr(0, pos);
            protected_groups.push_back(token != "0");
            pg.erase(0, pos + delim.length());
        }
        string msg =  "protected_groups: "; 
        for (auto pg : protected_groups)
            msg += pg + ",";
        msg += "\n";
        logger.log(msg,2);
    }
}
string Parameters::get_protected_groups()
{
    string out = "";
    for (int i = 0; i < protected_groups.size(); ++i)
    {
        out += protected_groups.at(i);
        if (i < protected_groups.size() - 1)
            out += ",";
    }
    return out;
}

void Parameters::set_feature_names(string fn)
{
    if (fn.empty())
        feature_names.clear();
    else
    {
        fn += ',';      // add delimiter to end
        string delim=",";
        size_t pos = 0;
        string token;
        while ((pos = fn.find(delim)) != string::npos) 
        {
            token = fn.substr(0, pos);
            feature_names.push_back(token);
            fn.erase(0, pos + delim.length());
        }
    }
}
string Parameters::get_feature_names()
{
    return ravel(this->feature_names);
}

void Parameters::set_functions(string fs)
{
    /*! 
     * Input: 
     *
     *		fs: string of comma-separated Node names
     *
     * Output:
     *
     *		modifies functions 
     *
     */
    this->function_str = fs;

    if (fs.empty())
        fs = "+,-,*,/,^2,^3,sqrt,sin,cos,exp,log,^,"
              "logit,tanh,gauss,relu,"
              "split,split_c,"
              "b2f,c2f,and,or,not,xor,=,<,<=,>,>=,if,ite";
    fs += ',';          // add delimiter to end 
    string delim = ",";
    size_t pos = 0;
    string token;
    this->functions.clear();
    while ((pos = fs.find(delim)) != string::npos) 
    {
        token = fs.substr(0, pos);

        functions.push_back(createNode(token));

        fs.erase(0, pos + delim.length());
    } 
    
    string log_msg = "functions set to [";
    for (const auto& f: functions) log_msg += f->name + ", "; 
    log_msg += "]\n";
    
    logger.log(log_msg, 3);
    
    // reset output types
    set_otypes();
}
string Parameters::get_functions_()
{
    vector<string> fn_vec;
    for (const auto& fn : this->functions)
        fn_vec.push_back(fn->name);
    return ravel(fn_vec);
}

void Parameters::set_op_weights()
{
    /*!
     * sets operator weights proportionately to the number of variables of 
     * each type that they operate on in the input data. 
     * depends on terminals already being set.
     */
    
    // 
    // first, count up the instances of each type of terminal. 
    // 
    int b_count = 0;
    int c_count = 0;
    int f_count = 0;
    int z_count = 0;
    int total_terms = 0;

    for (const auto& term : terminals)
    {
        switch (term->otype)
        {
            case 'b':
                ++b_count; 
                break;
            case 'c':
                ++c_count; 
                break;
            case 'f':
                ++f_count; 
                break;
            case 'z':
                ++z_count; 
                break;
        }
        ++total_terms;
    }
    // 
    // next, calculate the operator weights.
    // an operators weight is defined as 
    //      1/total_args * sum ([arg_type]_count/total_terminals) 
    // summed over each arg the operator takes
    //
    op_weights.clear();
    int i = 0;
    for (const auto& op : functions)
    {
        op_weights.push_back(0.0);
        int total_args = 0;
        for (auto& kv : op->arity) 
        {
            switch (kv.first) // kv.first is the arity type (character)
            {
                case 'b':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(b_count)/float(total_terms); 
                    break;
                case 'c':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(c_count)/float(total_terms); 
                    break;
                case 'f':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(f_count)/float(total_terms); 
                    break;
                case 'z':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(z_count)/float(total_terms); 
                    break;
            }
            total_args += kv.second;
        }
        op_weights.at(i) /= float(total_args);
        ++i;
    }
    // Now, we need to account for the output types of the operators that have non-zero 
    // weights, in addition to the terminals. 
    // So we now upweight the terminals according to the output types of the terminals that have
    // non-zero weights. 
    
    int total_ops_terms = total_terms;
    b_count = 0; 
    c_count = 0;
    f_count = 0;
    z_count = 0;
    for (unsigned i = 0; i < functions.size(); ++i)
    {
        if (op_weights.at(i) > 0) 
        {
            switch (functions.at(i)->otype)
            {
                case 'b':
                    ++b_count; 
                    break;
                case 'c':
                    ++c_count; 
                    break;
                case 'f':
                    ++f_count; 
                    break;
                case 'z':
                    ++z_count; 
                    break;
            }
        }
        ++total_ops_terms;
    }
    /* cout << "b_count: " << b_count << "\n" */
    /*      << "f_count: " << f_count << "\n" */
    /*      << "c_count: " << c_count << "\n" */
    /*      << "z_count: " << z_count << "\n" */
    /*      << "total_ops_terms: " << total_ops_terms << "\n"; */

    i = 0; // op_weights counter
    for (const auto& op : functions)
    {
        int total_args = 0;
        for (auto& kv : op->arity) 
        {
            switch (kv.first) // kv.first is the arity type (character)
            {
                case 'b':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(b_count)/float(total_ops_terms); 
                    break;
                case 'c':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(c_count)/float(total_ops_terms); 
                    break;
                case 'f':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(f_count)/float(total_ops_terms); 
                    break;
                case 'z':
                    for (unsigned j = 0; j < kv.second; ++j)
                        op_weights.at(i) += float(z_count)/float(total_ops_terms); 
                    break;
            }
            total_args += kv.second;
        }
        op_weights.at(i) /= float(total_args);
        ++i;
    }

    /* string ow = "op_weights: "; */
    /* for (unsigned i = 0; i< functions.size(); ++i) */
    /*     ow += "(" + functions.at(i)->name + ", " + std::to_string(op_weights.at(i)) + "), "; */ 
    /* ow += "\n"; */
    /* logger.log(ow,2); */
}

void Parameters::set_terminals(int nf, const LongData& Z)
{
    terminals.clear();
    num_features = nf; 
    for (size_t i = 0; i < nf; ++i)
        terminals.push_back(createNode(string("x"), 0, 0, i));
    
    if(erc)
    {
        for (int i = 0; i < nf; ++i)
        {
            if(r() < 0.5)
                terminals.push_back(createNode(string("kb"), 0, r(), 0));
            else
                terminals.push_back(createNode(string("kd"), r(), 0, 0));
        }        
    }

    for (const auto &val : Z)
    {
        longitudinalMap.push_back(val.first);
        terminals.push_back(createNode(string("z"), 0, 0, 0, val.first));
    }
    // reset output types
    set_ttypes();
    
    set_otypes(true);
    set_op_weights();
    // set dummy term_weights to zero
    this->set_term_weights(vector<float>());
}


void Parameters::set_objectives(string obj)
{
    /*! Input: obj, a comma-separated list of objectives
     */

    obj += ',';          // add delimiter to end 
    string delim = ",";
    size_t pos = 0;
    string token;
    objectives.clear();
    while ((pos = obj.find(delim)) != string::npos) 
    {
        token = obj.substr(0, pos);
        objectives.push_back(token);
        obj.erase(0, pos + delim.length());
    }
}
string Parameters::get_objectives()
{
    return ravel(this->objectives);
}


void Parameters::set_verbosity(int verbosity)
{
    logger.set_log_level(verbosity);
    this->verbosity = verbosity;
}

void Parameters::set_classes(const VectorXf& y)
{
    classes.clear();

    // set class labels
    vector<float> uc = unique(y);

    string str_classes = "{";
    for (auto c : uc)
        str_classes += to_string(c) + ",";
    str_classes = str_classes.substr(0,str_classes.size()-1);
    str_classes += "}";

    // check that class labels are contiguous and start at 0
    if (int(uc.at(0)) != 0)
        THROW_INVALID_ARGUMENT("Class labels must start at 0 and be "
                "contiguous. The input classes are " + str_classes);
    vector<int> cont_classes(uc.size());
    iota(cont_classes.begin(), cont_classes.end(), 0);
    for (int i = 0; i < cont_classes.size(); ++i)
    {
        if ( int(uc.at(i)) != cont_classes.at(i))
            THROW_INVALID_ARGUMENT("Class labels must start at 0 and be "
                    "contiguous. Passed labels = " + str_classes);
    }
    n_classes = uc.size();

    for (auto c : uc)
        classes.push_back(int(c));
}

void Parameters::set_sample_weights(VectorXf& y)
{
    // set class weights
    class_weights.resize(n_classes);
    sample_weights.clear();
    for (unsigned i = 0; i < n_classes; ++i){
        class_weights.at(i) = float(
               (y.cast<int>().array() == int(classes.at(i))).count())/y.size(); 
        class_weights.at(i) = (1 - class_weights.at(i))*float(n_classes);
    }
    for (unsigned i = 0; i < y.size(); ++i)
    {
        sample_weights.push_back(class_weights.at(int(y(i))));
    }
}
/* void Parameters::initialize_node_map() */
/* { */

/*    this->node_map = { */
/*         //arithmetic operators */
/*         { "+",  new NodeAdd({1.0,1.0})}, */ 
/*         { "-",  new NodeSubtract({1.0,1.0})}, */ 
/*         { "*",  new NodeMultiply({1.0,1.0})}, */ 
/*         { "/",  new NodeDivide({1.0,1.0})}, */ 
/*         { "sqrt",  new NodeSqrt({1.0})}, */ 
/*         { "sin",  new NodeSin({1.0})}, */ 
/*         { "cos",  new NodeCos({1.0})}, */ 
/*         { "tanh",  new NodeTanh({1.0})}, */ 
/*         { "^2",  new NodeSquare({1.0})}, */ 
/*         { "^3",  new NodeCube({1.0})}, */ 
/*         { "^",  new NodeExponent({1.0})}, */ 
/*         { "exp",  new NodeExponential({1.0})}, */ 
/*         { "gauss",  new NodeGaussian({1.0})}, */ 
/*         { "gauss2d",  new Node2dGaussian({1.0, 1.0})}, */ 
/*         { "log", new NodeLog({1.0}) }, */   
/*         { "logit", new NodeLogit({1.0}) }, */
/*         { "relu", new NodeRelu({1.0}) }, */
/*         { "b2f", new NodeFloat<bool>() }, */
/*         { "c2f", new NodeFloat<int>() }, */

/*         // logical operators */
/*         { "and", new NodeAnd() }, */
/*         { "or", new NodeOr() }, */
/*         { "not", new NodeNot() }, */
/*         { "xor", new NodeXor() }, */
/*         { "=", new NodeEqual() }, */
/*         { ">", new NodeGreaterThan() }, */
/*         { ">=", new NodeGEQ() }, */        
/*         { "<", new NodeLessThan() }, */
/*         { "<=", new NodeLEQ() }, */
/*         { "split", new NodeSplit<float>() }, */
/*         { "fuzzy_split", new NodeFuzzySplit<float>() }, */
/*         { "fuzzy_fixed_split", new NodeFuzzyFixedSplit<float>() }, */
/*         { "split_c", new NodeSplit<int>() }, */
/*         { "fuzzy_split_c", new NodeFuzzySplit<int>() }, */
/*         { "fuzzy_fixed_split_c", new NodeFuzzyFixedSplit<int>() }, */
/*         { "if", new NodeIf() }, */   	    		
/*         { "ite", new NodeIfThenElse() }, */
/*         { "step", new NodeStep() }, */
/*         { "sign", new NodeSign() }, */

/*         // longitudinal nodes */
/*         { "mean", new NodeMean() }, */
/*         { "median", new NodeMedian() }, */
/*         { "max", new NodeMax() }, */
/*         { "min", new NodeMin() }, */
/*         { "variance", new NodeVar() }, */
/*         { "skew", new NodeSkew() }, */
/*         { "kurtosis", new NodeKurtosis() }, */
/*         { "slope", new NodeSlope() }, */
/*         { "count", new NodeCount() }, */
/*         { "recent", new NodeRecent() }, */
/*         // terminals */
/*         { "variable_f", new NodeVariable<float>() }, */
/*         { "variable_b", new NodeVariable<bool>() }, */
/*         { "variable_c", new NodeVariable<int>() }, */
/*         { "constant_b", new NodeConstant(false) }, */
/*         { "constant_d", new NodeConstant(0.0) }, */
/*     }; */
/* } */
}
