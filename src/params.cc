/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "params.h"

namespace FT{

        
    Parameters::Parameters(int pop_size, int gens, string ml, bool classification, int max_stall, 
               char ot, int verbosity, string fs, float cr, unsigned int max_depth, 
               unsigned int max_dim, bool constant, string obj, bool sh, double sp, 
               double fb, string sc, string fn, bool bckprp, int iters, double lr,
               int bs, bool hclimb, int maxt, bool useb):    
            pop_size(pop_size),
            gens(gens),
            ml(ml),
            classification(classification),
            max_stall(max_stall), 
            cross_rate(cr),
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
            use_batch(useb)
        {
            set_verbosity(verbosity);
            if (fs.empty())
                fs = "+,-,*,/,^2,^3,sqrt,sin,cos,exp,log,^,"
                      "logit,tanh,gauss,relu,split,split_c,b2f,"
                      "c2f,and,or,not,xor,=,<,<=,>,>=,if,ite";
            set_functions(fs);
            set_objectives(obj);
            set_feature_names(fn);
            updateSize();     
            set_otypes();
            n_classes = 2;
            set_scorer(sc);
        }
    
    Parameters::~Parameters(){}
    
    /*! checks initial parameter settings before training.
     *  make sure ml choice is valid for problem type.
     *  make sure scorer is set. 
     *  for classification, check clases and find number.
     */
    void Parameters::init()
    {
        if (!ml.compare("LinearRidgeRegression") && classification)
        {
            msg("Setting ML type to LR",3);
            ml = "LR";            
        }
    }
    
    /// print message with verbosity control. 
    string Parameters::msg(string m, int v, string sep) const
    {
        /* prints messages based on verbosity level. */
		string msg = "";
        if (verbosity >= v)
        {
            std::cout << m << sep;
            msg += m+sep;
        }
        return msg;
    }
  
    /// sets current generation
    void Parameters::set_current_gen(int g) { current_gen = g; }
    
    /// sets scorer type
    void Parameters::set_scorer(string sc)
    {
        if (sc.empty())
        {
            if (classification && n_classes == 2)
            {
                if (ml.compare("LR") || ml.compare("SVM"))
                    scorer = "log";
                else
                    scorer = "zero_one";
            }
            else if (classification){
                if (ml.compare("LR") || ml.compare("SVM"))
                    scorer = "multi_log";
                else
                    scorer = "bal_zero_one";
            }
            else
                scorer = "mse";
        }
        else
            scorer = sc;
        msg("scorer set to " + scorer,3);
    }
    
    /// sets weights for terminals. 
    void Parameters::set_term_weights(const vector<double>& w)
    {           
        //assert(w.size()==terminals.size()); 
        /* cout << "weights: "; for (auto tmp : w) cout << tmp << " " ; cout << "\n"; */ 
        string weights;
        double u = 1.0/double(w.size());
        term_weights.clear();

        // take abs value of weights
        vector<double> aw = w;
        double sum = 0;
        for (unsigned i = 0; i < aw.size(); ++i)
        { 
            aw[i] = fabs(aw[i]); 
            sum += aw[i];
        }
        // softmax transform values
        /* vector<double> sw = softmax(aw); */
        /* cout << "sw: "; for (auto tmp : sw) cout << tmp << " " ; cout << "\n"; */ 
        // normalize weights to one
        for (unsigned i = 0; i < aw.size(); ++i)
        { 
            aw[i] = aw[i]/sum;
        }
        int x = 0;
        // assign transformed weights as terminal weights
        for (unsigned i = 0; i < terminals.size(); ++i)
        {
            if(terminals[i]->otype == 'z')
                term_weights.push_back(u);
            else
            {
                term_weights.push_back(u + feedback*(aw[x]-u));
                x++;
            }
        }
           
        weights = "term weights: ";
        for (auto tw : term_weights)
            weights += std::to_string(tw)+" ";
        weights += "\n";
        
        msg(weights, 3);
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
    
    void Parameters::set_verbosity(int verbosity)
    {
        if(verbosity <=3 && verbosity >=0)
            this->verbosity = verbosity;
        else
        {
            HANDLE_ERROR_NO_THROW("'" + std::to_string(verbosity) + "' is not a valid verbosity. Setting to default 2\n");
            HANDLE_ERROR_NO_THROW("Valid Values :\n\t0 - none\n\t1 - progress\n\t2 - minimal\n\t3 - all");
            this->verbosity = 2;
        }
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
                if (ttypes.size()==1 && ttypes[0]=='b')
                {
                    msg("otypes is size 1 and otypes[0]==b\nerasing functions...\n",2);
                    size_t n = functions.size();
                    for (vector<int>::size_type i =n-1; 
                         i != (std::vector<int>::size_type) -1; i--){
                        if (functions.at(i)->arity['f'] >0){
                            msg("erasing function " + functions.at(i)->name + "\n", 2);
                            functions.erase(functions.begin()+i);
                        }
                    }
                    
                    otype = 'b';
                    otypes.push_back('b');
                }           
                else
                {
                    otypes.push_back('b');
                    otypes.push_back('f');
                }
                
                //erasing categorical nodes if no categorical stack exists  
                if (terminals_set && !in(ttypes, 'c'))
                {
                    size_t n = functions.size();
                    for (vector<int>::size_type i =n-1; 
                         i != (std::vector<int>::size_type) -1; i--){
                        if (functions.at(i)->arity['c'] >0){
                            msg("erasing function " + functions.at(i)->name + "\n", 2);
                            functions.erase(functions.begin()+i);
                        }
                    }
                }         
                break;
            }
        }

    }
    
    std::unique_ptr<Node> Parameters::createNode(string str,
                                                 double d_val,
                                                 bool b_val,
                                                 size_t loc,
                                                 string name)
    {
        // algebraic operators
    	if (str.compare("+") == 0) 
    		return std::unique_ptr<Node>(new NodeAdd());
        
        else if (str.compare("-") == 0)
    		return std::unique_ptr<Node>(new NodeSubtract());

        else if (str.compare("*") == 0)
    		return std::unique_ptr<Node>(new NodeMultiply());

     	else if (str.compare("/") == 0)
    		return std::unique_ptr<Node>(new NodeDivide());

        else if (str.compare("sqrt") == 0)
    		return std::unique_ptr<Node>(new NodeSqrt());
    	
    	else if (str.compare("sin") == 0)
    		return std::unique_ptr<Node>(new NodeSin());
    		
    	else if (str.compare("cos") == 0)
    		return std::unique_ptr<Node>(new NodeCos());
    		
    	else if (str.compare("tanh")==0)
            return std::unique_ptr<Node>(new NodeTanh());
    	   
        else if (str.compare("^2") == 0)
    		return std::unique_ptr<Node>(new NodeSquare());
 	
        else if (str.compare("^3") == 0)
    		return std::unique_ptr<Node>(new NodeCube());
    	
        else if (str.compare("^") == 0)
    		return std::unique_ptr<Node>(new NodeExponent());

        else if (str.compare("exp") == 0)
    		return std::unique_ptr<Node>(new NodeExponential());
    		
    	else if (str.compare("gauss")==0)
            return std::unique_ptr<Node>(new NodeGaussian());
        
        else if (str.compare("gauss2d")==0)
            return std::unique_ptr<Node>(new Node2dGaussian());

        else if (str.compare("log") == 0)
    		return std::unique_ptr<Node>(new NodeLog());   
    		
    	else if (str.compare("logit")==0)
            return std::unique_ptr<Node>(new NodeLogit());

        else if (str.compare("relu")==0)
            return std::unique_ptr<Node>(new NodeRelu());

        else if (str.compare("b2f")==0)
                return std::unique_ptr<Node>(new NodeFloat(false));
        
        else if (str.compare("c2f")==0)
                return std::unique_ptr<Node>(new NodeFloat(true));

        // logical operators
        else if (str.compare("and") == 0)
    		return std::unique_ptr<Node>(new NodeAnd());
       
    	else if (str.compare("or") == 0)
    		return std::unique_ptr<Node>(new NodeOr());
   		
     	else if (str.compare("not") == 0)
    		return std::unique_ptr<Node>(new NodeNot());
    		
    	else if (str.compare("xor")==0)
            return std::unique_ptr<Node>(new NodeXor());
   		
    	else if (str.compare("=") == 0)
    		return std::unique_ptr<Node>(new NodeEqual());
    		
        else if (str.compare(">") == 0)
    		return std::unique_ptr<Node>(new NodeGreaterThan());

    	else if (str.compare(">=") == 0)
    		return std::unique_ptr<Node>(new NodeGEQ());        

    	else if (str.compare("<") == 0)
    		return std::unique_ptr<Node>(new NodeLessThan());
    	
    	else if (str.compare("<=") == 0)
    		return std::unique_ptr<Node>(new NodeLEQ());
 
        else if (str.compare("split") == 0)
  		    return std::unique_ptr<Node>(new NodeSplit<double>());
  		
  		else if (str.compare("split_c") == 0)
  		    return std::unique_ptr<Node>(new NodeSplit<int>());
    	
     	else if (str.compare("if") == 0)
    		return std::unique_ptr<Node>(new NodeIf());   	    		
        	
    	else if (str.compare("ite") == 0)
    		return std::unique_ptr<Node>(new NodeIfThenElse());
    		
    	else if (str.compare("step")==0)
            return std::unique_ptr<Node>(new NodeStep());
            
        else if (str.compare("sign")==0)
            return std::unique_ptr<Node>(new NodeSign());
           
        // longitudinal nodes
        else if (str.compare("mean")==0)
            return std::unique_ptr<Node>(new NodeMean());
            
        else if (str.compare("median")==0)
            return std::unique_ptr<Node>(new NodeMedian());
            
        else if (str.compare("max")==0)
            return std::unique_ptr<Node>(new NodeMax());
        
        else if (str.compare("min")==0)
            return std::unique_ptr<Node>(new NodeMin());
        
        else if (str.compare("variance")==0)
            return std::unique_ptr<Node>(new NodeVar());
            
        else if (str.compare("skew")==0)
            return std::unique_ptr<Node>(new NodeSkew());
            
        else if (str.compare("kurtosis")==0)
            return std::unique_ptr<Node>(new NodeKurtosis());
            
        else if (str.compare("slope")==0)
            return std::unique_ptr<Node>(new NodeSlope());
            
        else if (str.compare("count")==0)
            return std::unique_ptr<Node>(new NodeCount());

        // variables and constants
        else if (str.compare("x") == 0)
        { 
            if(dtypes.size() == 0)
            {
                if (feature_names.size() == 0)
                    return std::unique_ptr<Node>(new NodeVariable(loc));
                else
                    return std::unique_ptr<Node>(new NodeVariable(loc,'f', feature_names.at(loc)));
            }
            else if (feature_names.size() == 0)
                return std::unique_ptr<Node>(new NodeVariable(loc, dtypes[loc]));
            else
                return std::unique_ptr<Node>(new NodeVariable(loc, dtypes[loc], 
                                                              feature_names.at(loc)));
        }
            
        else if (str.compare("kb")==0)
            return std::unique_ptr<Node>(new NodeConstant(b_val));
            
        else if (str.compare("kd")==0)
            return std::unique_ptr<Node>(new NodeConstant(d_val));
            
        else if (str.compare("z")==0)
            return std::unique_ptr<Node>(new NodeLongitudinal(name));
        else
            HANDLE_ERROR_THROW("Error: no node named '" + str + "' exists."); 
        
        //TODO: add squashing functions, time delay functions, and stats functions
    	
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

        fs += ',';          // add delimiter to end 
        string delim = ",";
        size_t pos = 0;
        string token;
        functions.clear();
        while ((pos = fs.find(delim)) != string::npos) 
        {
            token = fs.substr(0, pos);
            functions.push_back(createNode(token));
            /* if(!token.compare("float") || !token.compare("split")) */
            /*     functions.push_back(createNode(token, 0, true)); */
            fs.erase(0, pos + delim.length());
        } 
        if (verbosity > 2){
            std::cout << "functions set to [";
            for (const auto& f: functions) std::cout << f->name << ", "; 
            std::cout << "]\n";
        }
        // reset output types
        set_otypes();
    }

    void Parameters::set_terminals(int nf,
                                   std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z)
    {
        /*!
         * based on number of features.
         */
        terminals.clear();
        num_features = nf; 
        for (size_t i = 0; i < nf; ++i)
            terminals.push_back(createNode(string("x"), 0, 0, i));
    	
        if(erc)
    		for (int i = 0; i < nf; ++i)
    		{
    			if(r() < 0.5)
    	       		terminals.push_back(createNode(string("kb"), 0, r(), 0));
    	       	else
    	       		terminals.push_back(createNode(string("kd"), r(), 0, 0));
    	    }        
       
        for (const auto &val : Z)
        {
            longitudinalMap.push_back(val.first);
            terminals.push_back(createNode(string("z"), 0, 0, 0, val.first));
        }
        /* for (const auto& t : terminals) */ 
        /*     cout << t->name << " " ; */
        /* cout << "\n"; */
        // reset output types
        set_ttypes();
        set_otypes(true);
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

    void Parameters::set_classes(VectorXd& y)
    {
        classes.clear();

        // set class labels
        vector<double> uc = unique(y);
        
        n_classes = uc.size();

        for (auto c : uc)
            classes.push_back(int(c));
    }

    void Parameters::set_sample_weights(VectorXd& y)
    {
        // set class weights
        /* cout << "setting sample weights\n"; */
        class_weights.resize(n_classes);
        sample_weights.clear();
        for (unsigned i = 0; i < n_classes; ++i){
            class_weights.at(i) = float((y.cast<int>().array() == int(classes.at(i))).count())/y.size(); 
            class_weights.at(i) = (1 - class_weights.at(i))*float(n_classes);
        }
        /* cout << "y size: " << y.size() << "\n"; */
        for (unsigned i = 0; i < y.size(); ++i)
            sample_weights.push_back(class_weights.at(int(y(i))));
        /* std::cout << "sample weights size: " << sample_weights.size() << "\n"; */
        /* std::cout << "class weights: "; */ 
        /* for (auto c : class_weights) std::cout << c << " " ; std::cout << "\n"; */
        /* std::cout << "number of classes: " << n_classes << "\n"; */
    }
}
