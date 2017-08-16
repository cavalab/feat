#include "fewtwo.h"
using FT::Fewtwo;

int main(int argc, char** argv){
    // runs FEWTWO from the command line. 
    Fewtwo fewtwo; 
    
    std::cout << "hello i'm FEWTWO!\n";

    MatrixXd X; 
    VectorXd y; 

    std::cout<< "fitting model...\n";
    
    fewtwo.fit(X,y);

    std::cout << "done!\n";

    return 0;

}


