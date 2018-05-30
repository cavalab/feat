/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H
// internal includes
#include "nodewrapper.h"
#include "nodevector.h"

namespace FT{

    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Parameters
     * @brief holds the hyperparameters for Feat. 
     */
    struct Parameters
    {
        int pop_size;                   			///< population size
        int gens;                       			///< max generations
        int current_gen;                            ///< holds current generation
        string ml;                      			///< machine learner used with Feat
        bool classification;            			///< flag to conduct classification rather than 
        int max_stall;                  			///< maximum stall in learning, in generations
        vector<char> otypes;                     	///< program output types ('f', 'b')
        vector<char> ttypes;                     	///< program terminal types ('f', 'b')
        char otype;                                 ///< user parameter for output type setup
        int verbosity;                  			///< amount of printing. 0: none, 1: minimal, 
                                                    // 2: all
        vector<double> term_weights;    			///< probability weighting of terminals
        NodeVector functions;                       ///< function nodes available in programs
        NodeVector terminals;                       ///< terminal nodes available in programs
        vector<std::string> longitudinalMap;        ///<vector storing longitudinal data keys

        unsigned int max_depth;         			///< max depth of programs
        unsigned int max_size;          			///< max size of programs (length)
        unsigned int max_dim;           			///< maximum dimensionality of programs
        bool erc;								    ///<whether to include constants for terminals 
        unsigned num_features;                      ///< number of features
        vector<string> objectives;                  ///< Pareto objectives 
        bool shuffle;                               ///< option to shuffle the data
        double split;                               ///< fraction of data to use for training
        vector<char> dtypes;                        ///< data types of input parameters
        double feedback;                            ///< strength of ml feedback on probabilities
        unsigned int n_classes;                     ///< number of classes for classification 
        float cross_rate;                           ///< cross rate for variation
        vector<int> classes;                        ///< class labels
        vector<float> class_weights;                ///< weights for each class
        vector<float> sample_weights;               ///< weights for each sample 
        string scorer;                              ///< loss function
        vector<string> feature_names;               ///< names of features
        bool backprop;                              ///< turns on backpropagation
        bool hillclimb;                             ///< turns on parameter hill climbing

        struct BP 
        {
           int iters;
           double learning_rate;
           int batch_size;
           BP(int i, double l, int bs): iters(i), learning_rate(l), batch_size(bs) {}
        };

        BP bp;                                      ///< backprop parameters
        
        struct HC 
        {
           int iters;
           double step;
           HC(int i, double s): iters(i), step(s) {}
        };
        
        HC hc;                                      ///< stochastic hill climbing parameters       
        
        Parameters(int pop_size, int gens, string ml, bool classification, int max_stall, 
                   char ot, int verbosity, string fs, float cr, unsigned int max_depth, 
                   unsigned int max_dim, bool constant, string obj, bool sh, double sp, 
                   double fb, string sc, string fn, bool bckprp, int iters, double lr,
                   int bs, bool hclimb):    
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
            hc(iters, lr)
        {
            set_verbosity(verbosity);
            if (fs.empty())
                fs = "+,-,*,/,^2,^3,sqrt,sin,cos,exp,log,^,"
                      "logit,tanh,gauss,relu,split,"
                      "and,or,not,xor,=,<,<=,>,>=,if,ite";
            set_functions(fs);
            set_objectives(obj);
            set_feature_names(fn);
            updateSize();     
            set_otypes();
            n_classes = 2;
            set_scorer(sc);
        }
        
        ~Parameters(){}
        
        /*! checks initial parameter settings before training.
         *  make sure ml choice is valid for problem type.
         *  make sure scorer is set. 
         *  for classification, check clases and find number.
         */
        void init()
        {
            if (!ml.compare("LinearRidgeRegression") && classification)
            {
                msg("Setting ML type to LR",2);
                ml = "LR";            
            }


        }
        /// print message with verbosity control. 
        string msg(string m, int v, string sep="\n") const
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
        void set_current_gen(int g) { current_gen = g; }
        /// sets scorer type
        void set_scorer(string sc)
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
            msg("scorer set to " + scorer,2);
        }
        /// sets weights for terminals. 
        void set_term_weights(const vector<double>& w)
        {           
            //assert(w.size()==terminals.size()); 
            string weights;
            double u = 1.0/double(w.size());
            term_weights.clear();
            vector<double> sw = softmax(w);
            int x = 0;
            for (unsigned i = 0; i < terminals.size(); ++i)
            {
                if(terminals[i]->otype == 'z')
                    term_weights.push_back(u);
                else
                {
                    term_weights.push_back(u + feedback*(sw[x]-u));
                    x++;
                }
            }
                
            weights = "term weights: ";
            for (auto tw : term_weights)
                weights += std::to_string(tw)+" ";
            weights += "\n";
            
            msg(weights, 2);
        }
        
        /// return shared pointer to a node based on the string passed
        std::unique_ptr<Node> createNode(std::string str, double d_val = 0, bool b_val = false, 
                                         size_t loc = 0, string name = "");
        
        /// sets available functions based on comma-separated list.
        void set_functions(string fs);
        
        /// max_size is max_dim binary trees of max_depth
        void updateSize()
        {
        	max_size = (pow(2,max_depth+1)-1)*max_dim;
        }
        
        /// set max depth of programs
        void set_max_depth(unsigned int max_depth)
        {
        	this->max_depth = max_depth;
        	updateSize();
        }
        
        /// set maximum dimensionality of programs
        void set_max_dim(unsigned int max_dim)
        {
        	this->max_dim = max_dim;
        	updateSize();
        }
        
        /// set the terminals
        void set_terminals(int nf,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > ());

        void set_feature_names(string fn); 
        /// set the objectives
        void set_objectives(string obj);
        
        /// set level of debug info
        void set_verbosity(int verbosity)
        {
            if(verbosity <=2 && verbosity >=0)
                this->verbosity = verbosity;
            else
            {
                HANDLE_ERROR_NO_THROW("'" + std::to_string(verbosity) + "' is not a valid verbosity. Setting to default 1\n");
                HANDLE_ERROR_NO_THROW("Valid Values :\n\t0 - none\n\t1 - minimal\n\t2 - all\n");
                this->verbosity = 1;
            }
        } 

        void set_otype(char ot){ otype = ot; set_otypes();}
        
        void set_ttypes()
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
        void set_otypes()
        {
            otypes.clear();
            // set output types
            switch (otype)
            { 
                case 'b': otypes.push_back('b'); break;
                case 'f': otypes.push_back('f'); break;
                default: 
                {
                    // if terminals are all boolean, remove floating point functions
                    if (ttypes.size()==1 && ttypes[0]=='b')
                    {
                        std::cout << "otypes is size 1 and otypes[0]==b\nerasing functions...\n";
                        size_t n = functions.size();
                        for (vector<int>::size_type i =n-1; 
                             i != (std::vector<int>::size_type) -1; i--){
                            if (functions.at(i)->arity['f'] >0){
                                std::cout << "erasing function " << functions.at(i)->name << "\n";
                                functions.erase(functions.begin()+i);
                            }
                        }
                        std::cout << "functions:\n";
                        for (const auto& f : functions)
                            std::cout << f->name << " "; 
                        std::cout << "\n";
                        otype = 'b';
                        otypes.push_back('b');
                    }                        
                    else
                    {
                        otypes.push_back('b');
                        otypes.push_back('f');
                    }
                    break;
                }
            }

        }
        /// sets the number of classes based on target vector y.
        void set_classes(VectorXd& y);
        void set_sample_weights(VectorXd& y);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    std::unique_ptr<Node> Parameters::createNode(string str, double d_val, bool b_val, size_t loc, string name)
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
    		return std::unique_ptr<Node>(new NodeSplit());
    	
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
        {
            //std::cout<<"******CALLED with name "<<name<<"\n";
            return std::unique_ptr<Node>(new NodeLongitudinal(name));
        }
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
            fs.erase(0, pos + delim.length());
        } 
        if (verbosity > 1){
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
        set_otypes();
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
#endif
