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
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
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
    
    Fewtwo fewtwo(100); 
    //fewtwo.set_functions("+,-");
    fewtwo.set_max_depth(2);
    fewtwo.set_max_dim(10);
    fewtwo.set_verbosity(1);
    
    //////////////////////////////////////// parse arguments
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h")){
        // Print help and exit. 
        cout << "Fewtwo is a feature engineering wrapper for learning intelligible models.\n";
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
        fewtwo.set_classification(bool(stoi(input.getCmdOption("-class"))));
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




    
    
    ///////////////////////////////////////
    cout << "hello i'm FEWTWO!\n";
    
    // x1 = sin(t), x2 = cos(t), t = 0, 0.5, 1, 1.5, 2, 2.5, 3
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

    cout << "X shape: " << X.rows() << "x" << X.cols() << "\n";
    cout << "y shape: " << y.size() << "\n";

    cout<< "initializing model...\n";
    

    cout << "fitting model...\n";

    fewtwo.fit(X,y);

    cout << "done!\n";

    return 0;

}


