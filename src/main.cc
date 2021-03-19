#include <stdio.h>
#include "feat.h"

using FT::Feat;
#include <Eigen/Dense>
#include <shogun/base/init.h>
using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::string;
using std::stoi;
using std::to_string;
using std::stof;
using std::cout;
using namespace shogun;
// Command line parser
class InputParser{
    public:
        std::string dataset;
        
        InputParser (int &argc, char **argv){
            int start = 1;
            if (argc < 2){
                dataset="";
            }
            else{
                //unless help message is requested
                if (std::string(argv[1]).compare("-h"))                 {
                    dataset = argv[1];
                    start = 2;
                }            
                for (int i=start; i < argc; ++i)
                    this->tokens.push_back(std::string(argv[i]));            
            }
        }
        /// returns the value of a given command option 
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// checks whether a command option exists
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
        
    private:
        std::vector <std::string> tokens;
         
};

int main(int argc, char** argv){
    // runs FEAT from the command line.     
    
    Feat feat;
    std::string sep = ",";
    std::string ldataFile = "";
    float split = 0.75;    // split of input data used to trian Feat

    //////////////////////////////////////// parse arguments
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h") || input.dataset.empty()){
        if (input.dataset.empty() && !input.cmdOptionExists("-h")) 
            WARN("Error: no dataset specified.\n---\n");
        // Print help and exit. 
        cout << "Feat is a feature engineering wrapper for learning intelligible models.\n";
        cout << "Usage:\tfeat path/to/dataset [options]\n";
        cout << "Options\tDescription (default value)\n";
        cout << "-p\tpopulation size (100)\n";
        cout << "-g\tgenerations (100)\n";
        cout << "-ml\tMachine learning model pairing " \
                "(LinearRidgeRegression or LogisticRegression)\n";
        cout << "--c\tDo classification instead of regression. (false)\n";
        cout << "-v\tVerbosity. 0: none; 1: stats; 2: debugging (1)\n";
        cout << "-stall\tMaximum generations with no improvements to best score. (0, off)\n";
        cout << "-sel\tSelection method. (lexicase)\n";
        cout << "-surv\tSurvival method. (nsga2)\n";
        cout << "-xr\tCrossover rate in [0, 1]. Mutation is the reciprocal. (0.5)\n";
        cout << "-root_xr\tRoot crossover rate in [0, 1]. Subtree crossover is the reciprocal. (0.5)\n";
        cout << "-ops\tComma-separated list of functions to use. (all)\n";
        cout << "-depth\tMaximum feature depth. (3)\n";
        cout << "-dim\tMaximum program dimensionality. (10)\n";
        cout << "-r\tSet random seed. (set randomly)\n";
        cout << "-sep\tInput file separator / delimiter. Choices: , or ""\\\\t"" for tab (,)\n";
        cout << "--no-shuffle\tDon't shuffle data before splitting into train/validate sets."
                "(false)\n";
        cout << "-split\tFraction of data to use for training (0.75)\n";
        cout << "-isplit\tInternal slit for Feat's training procedure (0.75)\n";
        cout << "-f\tfeedback strength of ML on variation probabilities (0.5)\n";
        cout << "-log\tlog file name\n";
        cout << "-n_jobs\tmaximum number of threads\n";
        cout << "-ldata\tpath to longitudinal data file\n";
        cout << "-scorer\tscoring function [mse, zero_one, bal_zero_one, log, multi_log]\n"; 
        cout << "-bp\tbackpropagation iterations (zero)\n"; 
        cout << "-hc\tstochastic hill climbing iterations (zero)\n"; 
        cout << "-lr\tbackpropagation learning rate or hill climbing step size(zero)\n"; 
        cout << "-batch\tminibatch size (off if not set, default 100)\n"; 
        cout << "-max_time\tMaximum time in seconds to fit the model." \
                " Used in conjunction with generation limit.\n";
        cout << "--use_batch\tSet flag for stochastic mini batch training\n";
        cout << "-otype\tSet output types of features. 'b':bool only,'f':float only,'a':all\n";
        cout << "-obj\tComma-separated objectives. Choices: fitness, complexity, size, CN, corr" \
                " (size,complexity)\n";
        cout << "--residual_xo\tSet flag for residual crossover\n";
        cout << "--stagewise_xo\tSet flag for stagewise crossover\n";
        cout << "--softmax\tSet flag to use softmax normalization of feedback\n";
        cout << "--simplify\tPost-run simplification\n";
        cout << "--corr_delete_mutate\tPost-run simplification\n";
        cout << "-save_pop\tPrint the population objective scores. 0: never, 1: at end, "
                "2: each generation. (0)\n";
        cout << "-starting_pop\tSpecify a filename containg json formatted starting population"
                "2: each generation. (0)\n";
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
    //cout << "reading inputs ...";
    if(input.cmdOptionExists("-p"))
        feat.set_pop_size(stoi(input.getCmdOption("-p")));
    if(input.cmdOptionExists("-g"))
        feat.set_gens(stoi(input.getCmdOption("-g")));
    if(input.cmdOptionExists("-ml"))
        feat.set_ml(input.getCmdOption("-ml"));
    if(input.cmdOptionExists("--c"))
        feat.set_classification(true);
    if(input.cmdOptionExists("-v"))
        feat.set_verbosity(stoi(input.getCmdOption("-v")));
    if(input.cmdOptionExists("-stall"))
        feat.set_max_stall(stoi(input.getCmdOption("-stall")));
    if(input.cmdOptionExists("-sel"))
        feat.set_selection(input.getCmdOption("-sel"));
    if(input.cmdOptionExists("-surv"))
        feat.set_survival(input.getCmdOption("-surv"));
    if(input.cmdOptionExists("-xr"))
        feat.set_cross_rate(stof(input.getCmdOption("-xr")));
    if(input.cmdOptionExists("-root_xr"))
        feat.set_root_xo_rate(stof(input.getCmdOption("-root_xr")));
    if(input.cmdOptionExists("-ops"))
        feat.set_functions(input.getCmdOption("-ops"));
    if(input.cmdOptionExists("-depth"))
        feat.set_max_depth(stoi(input.getCmdOption("-depth")));
    if(input.cmdOptionExists("-dim"))
    {
        string tmp = input.getCmdOption("-dim");
        if (!tmp.substr(tmp.length()-1).compare("x") || 
                !tmp.substr(tmp.length()-1).compare("X"))
            feat.set_max_dim(tmp);
        else
            feat.set_max_dim(stoi(tmp));
    }
    if(input.cmdOptionExists("-r"))
        feat.set_random_state(stoi(input.getCmdOption("-r")));
    if(input.cmdOptionExists("-sep")) // separator
        sep = input.getCmdOption("-sep");   
    if(input.cmdOptionExists("--no-shuffle"))
        feat.set_shuffle(false);
    if(input.cmdOptionExists("-split"))
        split = std::stod(input.getCmdOption("-split"));
    if(input.cmdOptionExists("-isplit"))
        feat.set_split(std::stod(input.getCmdOption("-isplit")));
    if(input.cmdOptionExists("-f"))
        feat.set_fb(std::stod(input.getCmdOption("-f")));
    if(input.cmdOptionExists("-log"))
        feat.set_logfile(input.getCmdOption("-log"));
    if(input.cmdOptionExists("-ldata"))
        ldataFile = input.getCmdOption("-ldata");
    if(input.cmdOptionExists("-scorer"))
        feat.set_scorer(input.getCmdOption("-scorer"));
    if(input.cmdOptionExists("-bp"))
    {
        feat.set_backprop(true);
        feat.set_iters(std::stoi(input.getCmdOption("-bp")));
    }
    if(input.cmdOptionExists("-hc"))
    {
        feat.set_hillclimb(true);
        feat.set_iters(std::stoi(input.getCmdOption("-hc")));
    }   
    if(input.cmdOptionExists("-lr"))
        feat.set_lr(std::stod(input.getCmdOption("-lr")));
    if(input.cmdOptionExists("-batch"))
    {
        feat.set_use_batch();
        feat.set_batch_size(std::stoi(input.getCmdOption("-batch")));
    }
    if(input.cmdOptionExists("-n_jobs"))
        feat.set_n_jobs(std::stoi(input.getCmdOption("-n_jobs")));
    if(input.cmdOptionExists("-max_time"))
    {
        int time = std::stoi(input.getCmdOption("-max_time"));
        if(time <= 0)
            WARN("WARNING: max_time cannot be less than equal to 0");
        else
            feat.set_max_time(time);
    }
    if(input.cmdOptionExists("--use_batch"))
        feat.set_use_batch();
    if(input.cmdOptionExists("-otype"))
        feat.set_otype(input.getCmdOption("-otype")[0]);
    if(input.cmdOptionExists("-obj"))
        feat.set_objectives(input.getCmdOption("-obj"));
    if(input.cmdOptionExists("--residual_xo"))
        feat.set_residual_xo();
    if(input.cmdOptionExists("--stagewise_xo"))
        feat.set_stagewise_xo();
    if(input.cmdOptionExists("--softmax"))
        feat.set_softmax_norm();
    if(input.cmdOptionExists("--simplify"))
        feat.set_simplify(true);
    if(input.cmdOptionExists("--corr_delete_mutate"))
        feat.set_corr_delete_mutate(true);
    if(input.cmdOptionExists("-save_pop"))
        feat.set_save_pop(std::stoi(input.getCmdOption("-save_pop")));
    if(input.cmdOptionExists("-starting_pop"))
        feat.set_starting_pop(input.getCmdOption("-starting_pop"));

    //cout << "done.\n";
    ///////////////////////////////////////

    // read in dataset
    char delim;
    if (!sep.compare("\\t")) delim = '\t';
    else if (!sep.compare(",")) delim = ',';
    else delim = sep[0];
    
    MatrixXf X;
    VectorXf y; 
    vector<string> names;
    vector<char> dtypes;
    bool binary_endpoint=false;
    
    cout << "load_csv...";
    FT::load_csv(input.dataset,X,y,names,dtypes,binary_endpoint,delim);
    feat.set_feature_names(FT::Util::ravel(names));
    feat.set_dtypes(dtypes);
    
    if (binary_endpoint)
    {
        if (!feat.get_classification())
            WARN("WARNING: binary endpoint detected. " \
                                  "Feat is set for regression.");
        else
            std::cout << "setting binary endpoint\n";
                      
    }
    
    LongData Z;
   
    if(ldataFile.compare("")) 
        FT::load_longitudinal(ldataFile, Z);
   
    if (split < 1.0)
    {
        /* split data into training and test sets */
        FT::DataRef d(X, y, Z, feat.get_classification());
        
        d.train_test_split(feat.get_shuffle(), split);
       
        MatrixXf X_tcopy = d.t->X;     
        LongData Z_tcopy = d.t->Z;
        VectorXf y_tcopy = d.t->y;
        MatrixXf X_vcopy = d.v->X;     
        LongData Z_vcopy = d.v->Z;
        VectorXf y_vcopy = d.v->y;

        cout << "fitting model...\n";
        
        feat.fit(d.t->X, d.t->y, d.t->Z);

        cout << "\ngenerating training prediction...\n";

        VectorXf yhat = feat.predict(X_tcopy,Z_tcopy);
        /* float score_t = feat.score(X_tcopy, y_tcopy, Z_tcopy); */
        float score_t = (yhat.array() - y_tcopy.array()).pow(2).mean();
        /* cout << "tmp score: " << tmp << "\n"; */
        /* VectorXf yhat = feat.predict(X_tcopy,Z_tcopy); */
        
        cout.precision(4);
        cout << "train score: " << score_t << "\n";
        
        if (!feat.get_classification())
        {
            float SSres = (y_tcopy-yhat).array().pow(2).sum() ;
            float SStot =  (y_tcopy.array() - y_tcopy.mean()).array().pow(2).sum();
            float r2t = (1 - SSres/SStot );
            cout << "train r2: " << r2t << "\n";
        }
        
        cout << "generating test prediction...\n";
        
        VectorXf yhatv = feat.predict(X_vcopy,Z_vcopy);
        float score = feat.score(d.v->X,d.v->y,d.v->Z);
        cout << "test score: " << score << "\n";
        
        if (!feat.get_classification())
        {
            float SSresv = (y_vcopy-yhatv).array().pow(2).sum() ;
            float SStotv =  (y_vcopy.array() - y_vcopy.mean()).array().pow(2).sum();
            float r2v = (1 - SSresv/SStotv );
            cout << "test r2: " << r2v << "\n"; 
        }
    }
    else
    {
        cout << "fitting model...\n";
        
        feat.fit(X, y, Z);
        
    }
    cout << "printing final model\n";
    cout << feat.get_model();
    cout << "done!\n";
	
    return 0;

}

