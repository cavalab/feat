#include <stdio.h>
#include "fewtwo.h"
#include "fewtwocv.h"
using FT::Fewtwo;
using FT::FewtwoCV;
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
    // runs FEWTWO from the command line.     
    
    Fewtwo fewtwo;
    std::string sep = ",";
    double split = 0.75;    // split of input data used to trian Fewtwo

    cout << "\n" << 
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n" << 
    "                                        FEWTWO                                               "
    << "\n" <<
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n";
 
    //////////////////////////////////////// parse arguments
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h") || input.dataset.empty()){
        if (input.dataset.empty()) std::cerr << "Error: no dataset specified.\n---\n";
        // Print help and exit. 
        cout << "Fewtwo is a feature engineering wrapper for learning intelligible models.\n";
        cout << "Usage:\tfewtwo path/to/dataset [options]\n";
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
        cout << "-isplit\tInternal slit for Fewtwo's training procedure (0.75)\n";
        cout << "-f\tfeedback strength of ML on variation probabilities (0.5)\n";
        cout << "-n\tname to append to files\n";
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
    //cout << "reading inputs ...";
    if(input.cmdOptionExists("-p"))
        fewtwo.set_pop_size(stoi(input.getCmdOption("-p")));
    if(input.cmdOptionExists("-g"))
        fewtwo.set_generations(stoi(input.getCmdOption("-g")));
    if(input.cmdOptionExists("-ml"))
        fewtwo.set_ml(input.getCmdOption("-ml"));
    if(input.cmdOptionExists("--c"))
        fewtwo.set_classification(true);
    if(input.cmdOptionExists("-v"))
        fewtwo.set_verbosity(stoi(input.getCmdOption("-v")));
    if(input.cmdOptionExists("-stall"))
        fewtwo.set_max_stall(stoi(input.getCmdOption("-stall")));
    if(input.cmdOptionExists("-sel"))
        fewtwo.set_selection(input.getCmdOption("-sel"));
    if(input.cmdOptionExists("-surv"))
        fewtwo.set_survival(input.getCmdOption("-surv"));
    if(input.cmdOptionExists("-xr"))
        fewtwo.set_cross_rate(stof(input.getCmdOption("-xr")));
    if(input.cmdOptionExists("-ops"))
        fewtwo.set_functions(input.getCmdOption("-ops"));
    if(input.cmdOptionExists("-depth"))
        fewtwo.set_max_depth(stoi(input.getCmdOption("-depth")));
    if(input.cmdOptionExists("-dim"))
    {
        string tmp = input.getCmdOption("-dim");
        if (!tmp.substr(tmp.length()-1).compare("x") || !tmp.substr(tmp.length()-1).compare("X"))
            fewtwo.set_max_dim(tmp);
        else
            fewtwo.set_max_dim(stoi(tmp));
    }
    if(input.cmdOptionExists("-r"))
        fewtwo.set_random_state(stoi(input.getCmdOption("-r")));
    if(input.cmdOptionExists("-sep")) // separator
        sep = input.getCmdOption("-sep");   
    if(input.cmdOptionExists("--shuffle"))
        fewtwo.set_shuffle(true);
    if(input.cmdOptionExists("-split"))
        split = std::stod(input.getCmdOption("-split"));
    if(input.cmdOptionExists("-isplit"))
        fewtwo.set_split(std::stod(input.getCmdOption("-isplit")));
    if(input.cmdOptionExists("-f"))
        fewtwo.set_feedback(std::stod(input.getCmdOption("-f")));
    if(input.cmdOptionExists("-n"))
        fewtwo.set_name(input.getCmdOption("-n"));
    
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
    fewtwo.set_dtypes(dtypes);
    
    if (binary_endpoint)
    {
        if (!fewtwo.get_classification())
            std::cerr << "WARNING: binary endpoint detected. Fewtwo is set for regression.";
        else
            std::cout << "setting binary endpoint\n";
                      
    }
     
    // split data into training and test sets
    MatrixXd X_t(X.rows(),int(X.cols()*split));
    MatrixXd X_v(X.rows(),int(X.cols()*(1-split)));
    VectorXd y_t(int(y.size()*split)), y_v(int(y.size()*(1-split)));
    FT::train_test_split(X,y,X_t,X_v,y_t,y_v,fewtwo.get_shuffle());      
    
         

    cout << "fitting model...\n";
    
    fewtwo.fit(X_t,y_t);

    cout << "generating prediction...\n";

    double score = fewtwo.score(X_v,y_v);
    
    cout << "test score: " << score << "\n";
    // write validation score to file
    std::ofstream out_score; 
    out_score.open("score_" + fewtwo.get_name() + ".txt");
    out_score << score ;
    out_score.close();
    // write validation score to file
    std::ofstream out_arc; 
    out_score.open("arc_" + fewtwo.get_name() + ".txt");
    out_score << fewtwo.get_eqns() ;
    out_score.close();
    
    cout << "done!\n";
	
	
    return 0;

}


