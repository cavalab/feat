/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>

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
        
            Data(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z): X(X), y(y), Z(Z){}
            
            MatrixXd& X;
            VectorXd& y;
            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z;
    };
    
    class DataRef
    {
        private:
            bool oCreated;
            bool tCreated;
            bool vCreated;
        public:
            Data *o = NULL;
            Data *v = NULL;
            Data *t = NULL;
            
            DataRef();
            
            ~DataRef();
            
            DataRef(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z,
                    MatrixXd& X_t, VectorXd& y_t, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t,
                    MatrixXd& X_v, VectorXd& y_v, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v);
                    
            void setOriginalData(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z);
            
            void setOriginalData(Data *d);
            
            void setTrainingData(MatrixXd& X_t, VectorXd& y_t, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t);
            
            void setTrainingData(Data *d);
            
            void setValidationData(MatrixXd& X_v, VectorXd& y_v, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v);
            
            void setValidationData(Data *d);
                
    };
        
}



#endif
