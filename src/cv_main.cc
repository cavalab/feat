#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <iostream>

using namespace std;
#include "feat.h"
#include "featcv.h"
using FT::Feat;
using FT::FeatCV;
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

string readInputFile(string file)
{
    std::ifstream t(file);
    string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    
    return str;
}

int main(int argc, char** argv){

    std::string sep = ",";
    std::string infile = "";
    std::string hyper_params = "";
    
    cout << "\n" << 
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n" << 
    "                                        FEATCV                                               "
    << "\n" <<
    "/////////////////////////////////////////////////////////////////////////////////////////////"
    << "\n";
 
    //////////////////////////////////////// parse arguments
    InputParser input(argc, argv);
    if(input.cmdOptionExists("-h") || input.dataset.empty()){
        if (input.dataset.empty()) std::cerr << "Error: no dataset specified.\n---\n";
        // Print help and exit. 
        cout << "FeatCV is a cross validation model for Feat - a feature engineering wrapper for learning intelligible models.\n";
        cout << "Usage:\tfeat_cv path/to/dataset [options]\n";
        cout << "Options\tDescription (default value)\n";
        cout << "-sep\tInput file separator / delimiter. Choices: , or ""\\\\t"" for tab (,)\n";
        cout << "-infile\tInput file containing string for cross validation object\n";
        cout << "-h\tDisplay this help message and exit.\n";
        return 0;
    }
    cout << "reading inputs ...";
    if(input.cmdOptionExists("-sep")) // separator
        sep = input.getCmdOption("-sep");
    if(input.cmdOptionExists("-infile")) // separator
        infile = input.getCmdOption("-infile");
    cout << "done.\n";
    ///////////////////////////////////////

    // read in dataset
    cout << "sep: " << sep << "\n";
    char delim;
    if (!sep.compare("\\t")) delim = '\t';
    else if (!sep.compare(",")) delim = ',';
    else delim = sep[0];
    
    if(infile.compare(""))
        hyper_params = readInputFile(infile);
    else
        hyper_params = "[{\
                            'pop_size': (100, 500)\
                            'generations': (100, 200)\
                            'feedback': (0.2, 0.5, 0.8)\
                            'ml': (\"LinearRidgeRegression\", \"CART\")\
                            'cross_rate': (0.25, 0.5, 0.75)\
                         }\
                        ]";
                        
    cout<<"Hyper params are \n*****\n"<<hyper_params<<"\n***\n";
    
    MatrixXd X;
    VectorXd y; 
    vector<string> names;
    vector<char> dtypes;
    bool binary_endpoint=false;
    
    cout << "load_csv...";
    FT::load_csv(input.dataset,X,y,names,dtypes,binary_endpoint,delim);
    
    FeatCV validator(5, hyper_params);
    
    validator.fit(X, y);
	
    return 0;
    

}


