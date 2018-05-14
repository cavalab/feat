/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

#include <string>
#include <Eigen/Dense>
#include <vector>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;

//#include "node/node.h"
//external includes

namespace FT
{
    /*!
     * @class Data
     * @brief data holding X, y, and Z data
     */
    class Data
    {
        //Data(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z): X(X), y(y), Z(Z){}
        public:
        
            Data(const MatrixXd& X, const VectorXd& y, const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z): X(X), y(y), Z(Z){}
            
            const MatrixXd& X;
            const VectorXd& y;
            const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z;
    };
}



#endif
