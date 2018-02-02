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

//int main(int argc, char **argv){
//    InputParser input(argc, argv);
//    if(input.cmdOptionExists("-h")){
//        // Do stuff
//    }
//    const std::string &filename = input.getCmdOption("-f");
//    if (!filename.empty()){
//        // Do interesting things ...
//    }
//    return 0;
//}

int main(int argc, char** argv){
    // runs FEWTWO from the command line.     
    
    Fewtwo fewtwo;
    std::string sep = ",";
    
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
        cout << "-f\tfeedback strength of ML on variation probabilities (0.5)\n";
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
    cout << "reading inputs ...";
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
   // if(input.cmdOptionExists("-otype"))
   //     fewtwo.set_cross_rate(input.getCmdOption("-otype")[0]);
    if(input.cmdOptionExists("-ops"))
        fewtwo.set_functions(input.getCmdOption("-ops"));
    if(input.cmdOptionExists("-depth"))
        fewtwo.set_max_depth(stoi(input.getCmdOption("-depth")));
    if(input.cmdOptionExists("-dim"))
        fewtwo.set_max_dim(stoi(input.getCmdOption("-dim")));
    if(input.cmdOptionExists("-r"))
        fewtwo.set_random_state(stoi(input.getCmdOption("-r")));
    if(input.cmdOptionExists("-sep")) // separator
        sep = input.getCmdOption("-sep");   
    if(input.cmdOptionExists("--shuffle"))
        fewtwo.set_shuffle(true);
    if(input.cmdOptionExists("-split"))
        fewtwo.set_split(std::stod(input.getCmdOption("-split")));
    if(input.cmdOptionExists("-f"))
        fewtwo.set_feedback(std::stod(input.getCmdOption("-f")));
    cout << "done.\n";
    ///////////////////////////////////////

    // read in dataset
    cout << "sep: " << sep << "\n";
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
    if (binary_endpoint)
    {
        if (!fewtwo.get_classification())
            std::cerr << "WARNING: binary endpoint detected. Fewtwo is set for regression.";
        else
            std::cout << "setting binary endpoint\n";
                      
    }
    
    
    //vector<string> ml;
    //ml.push_back("LinearRidgeRegression");
    //
    //FewtwoCV validator(ml);
    //
    //validator.validate_data(X, y);
    
    
    
    fewtwo.set_dtypes(dtypes);
    
    cout << "fitting model...\n";
    
    fewtwo.fit(X,y);
    

    //cout << "done!\n";
	
	
    return 0;

}


