#include <stdio.h>
#include "fewtwo.h"
using FT::Fewtwo;
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::stoi;
using std::to_string;
using std::stof;
using std::cout;
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
        cout << "-p\tpopulation size\n";
        cout << "-g\tgenerations (iterations)\n";
        cout << "-ml\tMachine learning model pairing\n";
        cout << "--c\tDo classification instead of regression.\n";
        cout << "-v\tVerbosity. 0: none; 1: stats; 2: debugging\n";
        cout << "-stall\tMaximum generations with no improvements to best score.\n";
        cout << "-sel\tSelection method.\n";
        cout << "-surv\tSurvival method.\n";
        cout << "-xr\tCrossover rate in [0, 1]\n";
        cout << "-ops\tComma-separated list of functions to use.\n";
        cout << "-depth\tMaximum feature depth.\n";
        cout << "-dim\tMaximum program dimensionality.\n";
        cout << "-r\tSet random seed.\n";
        cout << "-sep\tInput file separator / delimiter. Choices: , or ""\\\\t"" for tab\n";
        cout << "--shuffle\tShuffle data before splitting into train/validate sets.\n";
        cout << "-split\tFraction of data to use for training (default: .75)\n";
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
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


    ///////////////////////////////////////

    // read in dataset
    char delim;
    if (!sep.compare("\\t")) delim = '\t';
    else if (!sep.compare(",")) delim = ',';
    else delim = sep[0];

    MatrixXd X;
    VectorXd y; 
    vector<string> names;
    FT::load_csv(input.dataset,X,y,names,delim);    

    cout << "fitting model...\n";

    fewtwo.fit(X,y);

    cout << "done!\n";

    return 0;

}


