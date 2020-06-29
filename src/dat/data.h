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
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::VectorXi;
using Eigen::Dynamic;
using Eigen::Map;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;
typedef std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>> LongData;
// internal includes
//#include "params.h"
#include "../util/utils.h"
//#include "node/node.h"
//external includes

namespace FT
{
    /**
    * @namespace FT::Dat
    * @brief namespace containing Data structures used in Feat
    */
    namespace Dat{
        /*!
         * @class Data
         * @brief data holding X, y, and Z data
         */
        class Data
        {
            public:
     
                MatrixXf& X; // n_features x n_samples matrix of features 
                VectorXf& y; // n_samples labels
                LongData& Z; // longitudinal features
                bool classification;
                bool validation; 
                vector<bool> protect; // protected subgroups of features

                Data(MatrixXf& X, VectorXf& y, LongData& Z, bool c = false, 
                        vector<bool> protect = vector<bool>());

                void set_validation(bool v=true);
                void set_protected_groups();
                
                /// select random subset of data for training weights.
                void get_batch(Data &db, int batch_size) const;
                // protect_levels stores the levels of protected factors in X.
                map<int,vector<float>> protect_levels;   
                vector<int> protected_groups;
                int group_intersections;
                vector<ArrayXb> cases;  // used to pre-process cases if there 
                                        // aren't that many group intersections
        };
        
        /* !
         * @class DataRef
         * @brief Holds training and validation splits of data, 
         * with pointers to each.
         * */
        class DataRef
        {
            private:
                bool oCreated;
                bool tCreated;
                bool vCreated;
                // training and validation data
                MatrixXf X_t;
                MatrixXf X_v;
                VectorXf y_t;
                VectorXf y_v;
                LongData Z_t;
                LongData Z_v;
                
                bool classification;
                /* vector<bool> protect;   // indicator of protected subgroups */

            public:
                Data *o = NULL;     //< pointer to original data
                Data *v = NULL;     //< pointer to validation data
                Data *t = NULL;     //< pointer to training data
                
                DataRef();
                
                ~DataRef();
                
        
                DataRef(MatrixXf& X, VectorXf& y, LongData& Z, bool c=false,
                        vector<bool> protect = vector<bool>());

                void init(MatrixXf& X, VectorXf& y, LongData& Z, 
                        bool c=false, vector<bool> protect = vector<bool>());

                void setOriginalData(MatrixXf& X, VectorXf& y, 
                        LongData& Z, bool c=false,
                        vector<bool> protect = vector<bool>());
                
                void setOriginalData(Data *d);
                
                void setTrainingData(MatrixXf& X_t, VectorXf& y_t, 
                                   LongData& Z_t,
                                   bool c = false, 
                                   vector<bool> protect = vector<bool>());
                
                void setTrainingData(Data *d, bool toDelete = false);
                
                void setValidationData(MatrixXf& X_v, VectorXf& y_v, 
                                   LongData& Z_v,
                                   bool c = false, 
                                   vector<bool> protect = vector<bool>());
                
                void setValidationData(Data *d);
                
                /// shuffles original data
                void shuffle_data();
                
                /// split classification data as stratas
                void split_stratified(float split);
                
                /// splits data into training and validation folds.
                void train_test_split(bool shuffle, float split);

                void split_longitudinal(
                            LongData&Z,
                            LongData&Z_t,
                            LongData&Z_v,
                            float split);
                            
                /// reordering utility for shuffling longitudinal data.
                void reorder_longitudinal(vector<ArrayXf> &vec1, 
                        const vector<int>& order); 

        };
    }
}

#endif
