/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ML_H
#define ML_H

//external includes
#include <shogun/base/some.h>                                                                       
#include <shogun/base/init.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/common.h>                                                                      
#include <shogun/labels/RegressionLabels.h>                                                         
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/Features.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/regression/svr/LibLinearRegression.h>
/* #include <shogun/classifier/svm/LibLinear.h> */
#include <shogun/ensemble/MeanRule.h>
#include <shogun/ensemble/MajorityVote.h>
#include <cmath>
#include <shogun/machine/LinearMulticlassMachine.h>
// internal includes
#include "shogun/MyCARTree.h"
#include "shogun/MulticlassLogisticRegression.h"
#include "shogun/MyMulticlassLibLinear.h"
#include "shogun/MyLibLinear.h"
#include "../params.h"

// stuff being used
using std::string;
using std::dynamic_pointer_cast;
using std::shared_ptr; 
using std::make_shared;
using std::cout;
namespace sh = shogun;
using sh::EProblemType; 
using sh::EProbHeuristicType;
using sh::CBinaryLabels;
using sh::CMulticlassLabels;
using sh::CLabels;

namespace FT{

    using namespace Util;
     
    /**
     * @namespace FT::Model
     * @brief namespace containing ML methods used in Feat
     */
    namespace Model{
    
        /*!
         * @class ML
         * @brief class that specifies the machine learning algorithm to pair with Feat. 
         */
        class ML 
        {
            public:
            	
                ML(const Parameters& params, bool norm=true);
                
                void init();
            
                ~ML();

                // return vector of weights for model. 
                vector<float> get_weights();
                
                // train ml model on X and return label object. 
                shared_ptr<CLabels> fit(MatrixXf& X, VectorXf& y, const Parameters& params, bool& pass,
                             const vector<char>& dtypes=vector<char>());

                // train ml model on X and return estimation y. 
                VectorXf fit_vector(MatrixXf& X, VectorXf& y, const Parameters& params, bool& pass,
                             const vector<char>& dtypes=vector<char>());

                // predict using a trained ML model, returning a label object. 
                shared_ptr<CLabels> predict(MatrixXf& X);
                
                // predict using a trained ML model, returning a vector of predictions. 
                VectorXf predict_vector(MatrixXf& X);
     
                // predict using a trained ML model, returning a vector of predictions. 
                ArrayXXf predict_proba(MatrixXf& X);
               
                /// utility function to convert CLabels types to VectorXd types. 
                VectorXf labels_to_vector(const shared_ptr<CLabels>& labels);

                /* VectorXd predict(MatrixXd& X); */
                // set data types (for tree-based methods)            
                void set_dtypes(const vector<char>& dtypes);

                shared_ptr<sh::CMachine> p_est;     ///< pointer to the ML object
                string ml_type;                     ///< user specified ML type
                sh::EProblemType prob_type;         ///< type of learning problem; binary, multiclass 
                                                    ///  or regression 
                Normalizer N;                       ///< normalization
                int max_train_time;                 ///< max seconds allowed for training
                bool normalize;                     ///< control whether ML normalizes its input before 
                                                    ///  training
                /* double get_bias(int i) */
                /* {   // get bias at feature i. only works with linear machines */
                /*     auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_bias(); */
                /*     if (i < tmp.size()) */
                /*         return tmp[i]; */
                /*     else */
                /*     { */
                /*         std::cerr << "ERROR: invalid location access in get_bias()\n"; */
                /*         throw; */
                /*     } */
                /* } */
        };
    /////////////////////////////////////////////////////////////////////////////////////// Definitions
    }
}


#endif
