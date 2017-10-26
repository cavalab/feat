#include "fewtwo.h"
using FT::Fewtwo;
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char** argv){
    // runs FEWTWO from the command line. 
        
    std::cout << "hello i'm FEWTWO!\n";
    
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

    std::cout << "X shape: " << X.rows() << "x" << X.cols() << "\n";
    std::cout << "y shape: " << y.size() << "\n";

    std::cout<< "initializing model...\n";
    

    Fewtwo fewtwo(100); 
    //fewtwo.set_functions("+,-");
    fewtwo.set_max_depth(2);
    fewtwo.set_max_dim(3);
    fewtwo.set_verbosity(1);
    std::cout << "fitting model...\n";

    fewtwo.fit(X,y);

    std::cout << "done!\n";

    return 0;

}


