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
using Eigen::VectorXi;
using Eigen::Dynamic;
using Eigen::Map;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;
// internal includes
//#include "params.h"
#include "../util/utils.h"
//#include "node/node.h"
//external includes

namespace FT
{
    namespace Dat{
        /*!
         * @class Data
         * @brief data holding X, y, and Z data
         */
        class Data
        {
            //Data(MatrixXd& X, VectorXd& y, std::map<string, 
            //std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z): X(X), y(y), Z(Z){}
            public:
     
                MatrixXd& X;
                VectorXd& y;
                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z;
                bool classification;
                bool validation; 

                Data(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, 
                        vector<ArrayXd>>>& Z, bool c = false);

                void set_validation(bool v=true);
                
                /// select random subset of data for training weights.
                void get_batch(Data &db, int batch_size) const;
        };
        
        /* !
         * @class DataRef
         * @brief Holds training and validation splits of data, with pointers to each.
         * */
        class DataRef
        {
            private:
                bool oCreated;
                bool tCreated;
                bool vCreated;
                // training and validation data
                MatrixXd X_t;
                MatrixXd X_v;
                VectorXd y_t;
                VectorXd y_v;
                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_t;
                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_v;

            public:
                Data *o = NULL;     //< pointer to original data
                Data *v = NULL;     //< pointer to validation data
                Data *t = NULL;     //< pointer to training data
                
                DataRef();
                
                ~DataRef();
                
        
                DataRef(MatrixXd& X, VectorXd& y, 
                                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z, 
                                 bool c=false);
                        
                void setOriginalData(MatrixXd& X, VectorXd& y, 
                        std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z, bool c=false);
                
                void setOriginalData(Data *d);
                
                void setTrainingData(MatrixXd& X_t, VectorXd& y_t, 
                                   std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t,
                                   bool c = false);
                
                void setTrainingData(Data *d, bool toDelete = false);
                
                void setValidationData(MatrixXd& X_v, VectorXd& y_v, 
                                   std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v,
                                   bool c = false);
                
                void setValidationData(Data *d);
               
                /// splits data into training and validation folds.
                void train_test_split(bool shuffle, double split);

                void split_longitudinal(
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                            double split);

        };
    }
}

#endif
