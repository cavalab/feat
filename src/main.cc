#include <stdio.h>
#include "feat.h"

using FT::Feat;
#include <Eigen/Dense>
#include <shogun/base/init.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;
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
                if (std::string(argv[1]).compare("-h")) //unless help message is requested
                {
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
    double split = 0.75;    // split of input data used to trian Feat

    cout << "\n" << 
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n" << 
    "                                        FEAT                                               "
    << "\n" <<
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n";
 
    //////////////////////////////////////// parse arguments
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h") || input.dataset.empty()){
        if (input.dataset.empty()) std::cerr << "Error: no dataset specified.\n---\n";
        // Print help and exit. 
        cout << "Feat is a feature engineering wrapper for learning intelligible models.\n";
        cout << "Usage:\tfeat path/to/dataset [options]\n";
        cout << "Options\tDescription (default value)\n";
        cout << "-p\tpopulation size (100)\n";
        cout << "-g\tgenerations (100)\n";
        cout << "-ml\tMachine learning model pairing (LinearRidgeRegression or LogisticRegression)\n";
        cout << "--c\tDo classification instead of regression. (false)\n";
        cout << "-v\tVerbosity. 0: none; 1: stats; 2: debugging (1)\n";
        cout << "-stall\tMaximum generations with no improvements to best score. (off)\n";
        cout << "-sel\tSelection method. (lexicase)\n";
        cout << "-surv\tSurvival method. (nsga2)\n";
        cout << "-xr\tCrossover rate in [0, 1]. Mutation is the reciprocal. (0.5)\n";
        cout << "-ops\tComma-separated list of functions to use. (all)\n";
        cout << "-depth\tMaximum feature depth. (3)\n";
        cout << "-dim\tMaximum program dimensionality. (10)\n";
        cout << "-r\tSet random seed. (set randomly)\n";
        cout << "-sep\tInput file separator / delimiter. Choices: , or ""\\\\t"" for tab (,)\n";
        cout << "--shuffle\tShuffle data before splitting into train/validate sets. (false)\n";
        cout << "-split\tFraction of data to use for training (0.75)\n";
        cout << "-isplit\tInternal slit for Feat's training procedure (0.75)\n";
        cout << "-f\tfeedback strength of ML on variation probabilities (0.5)\n";
        cout << "-n\tname to append to files\n";
        cout << "-ldata\tpath to longitudinal data file\n";
        cout << "-scorer\tscoring function [mse, accuracy, bal_accuracy, log]\n"; 
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
    //cout << "reading inputs ...";
    if(input.cmdOptionExists("-p"))
        feat.set_pop_size(stoi(input.getCmdOption("-p")));
    if(input.cmdOptionExists("-g"))
        feat.set_generations(stoi(input.getCmdOption("-g")));
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
    if(input.cmdOptionExists("-ops"))
        feat.set_functions(input.getCmdOption("-ops"));
    if(input.cmdOptionExists("-depth"))
        feat.set_max_depth(stoi(input.getCmdOption("-depth")));
    if(input.cmdOptionExists("-dim"))
    {
        string tmp = input.getCmdOption("-dim");
        if (!tmp.substr(tmp.length()-1).compare("x") || !tmp.substr(tmp.length()-1).compare("X"))
            feat.set_max_dim(tmp);
        else
            feat.set_max_dim(stoi(tmp));
    }
    if(input.cmdOptionExists("-r"))
        feat.set_random_state(stoi(input.getCmdOption("-r")));
    if(input.cmdOptionExists("-sep")) // separator
        sep = input.getCmdOption("-sep");   
    if(input.cmdOptionExists("--shuffle"))
        feat.set_shuffle(true);
    if(input.cmdOptionExists("-split"))
        split = std::stod(input.getCmdOption("-split"));
    if(input.cmdOptionExists("-isplit"))
        feat.set_split(std::stod(input.getCmdOption("-isplit")));
    if(input.cmdOptionExists("-f"))
        feat.set_feedback(std::stod(input.getCmdOption("-f")));
    if(input.cmdOptionExists("-n"))
        feat.set_name(input.getCmdOption("-n"));
    if(input.cmdOptionExists("-ldata"))
        ldataFile = input.getCmdOption("-ldata");
    if(input.cmdOptionExists("-scorer"))
        feat.set_scorer(input.getCmdOption("-scorer"));
    //cout << "done.\n";
    ///////////////////////////////////////

    // read in dataset
    char delim;
    if (!sep.compare("\\t")) delim = '\t';
    else if (!sep.compare(",")) delim = ',';
    else delim = sep[0];
    
    MatrixXd X;
    VectorXd y; 
    vector<string> names;
    vector<char> dtypes;
    bool binary_endpoint=false;
    
    cout << "load_csv...";
    FT::load_csv(input.dataset,X,y,names,dtypes,binary_endpoint,delim);
    //feat.set_dtypes(dtypes);
    
    if (binary_endpoint)
    {
        if (!feat.get_classification())
            std::cerr << "WARNING: binary endpoint detected. Feat is set for regression.";
        else
            std::cout << "setting binary endpoint\n";
                      
    }
     
    // split data into training and test sets
    MatrixXd X_t(X.rows(),int(X.cols()*split));
    MatrixXd X_v(X.rows(),int(X.cols()*(1-split)));
    VectorXd y_t(int(y.size()*split)), y_v(int(y.size()*(1-split)));
    
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z;
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_t;
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_v;
    
    if(ldataFile.compare(""))
    {
        FT::load_longitudinal(ldataFile, Z);
        FT::split_longitudinal(Z, Z_t, Z_v, split);
    }   
    
    FT::train_test_split(X,y,Z,X_t,X_v,y_t,y_v,Z_t,Z_v,feat.get_shuffle(), split);      
    
    MatrixXd X_tcopy = X_t;     
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_tcopy = Z_t;
  
    cout << "fitting model...\n";
    
    feat.fit(X_t, y_t, Z_t);

    cout << "generating training prediction...\n";

    double score_t = feat.score(X_tcopy,y_t, Z_tcopy);

    cout.precision(5);
    cout << "train score: " << score_t << "\n";
    
    cout << "generating test prediction...\n";
    double score = feat.score(X_v,y_v);
    cout << "test score: " << score << "\n";

    cout << "done!\n";
	
    return 0;

}


