#include "fewtwo.h"
using FT::Fewtwo;
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char** argv){
    // runs FEWTWO from the command line. 
        
    std::cout << "hello i'm FEWTWO!\n";

    MatrixXd X(4,2); 
    X << 0, 4,
         1, 3, 
         2, 5, 
         3, 2;

    VectorXd y(4); 
    // y = 2*x1+3*x2
    y << 12, 11, 19, 12;

    std::cout << "X shape: " << X.rows() << "x" << X.cols() << "\n";
    std::cout << "y shape: " << y.size() << "\n";

    std::cout<< "initializing model...\n";
    
    Fewtwo fewtwo; 
    std::cout << "fitting model...\n";

    fewtwo.fit(X,y);

    std::cout << "done!\n";

    return 0;

}


