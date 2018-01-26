/* FEWTWO
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
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/ensemble/MajorityVote.h>
#include <cmath>
// internal includes
#include "ml/MyCARTree.h"


// stuff being used
using std::string;
using std::dynamic_pointer_cast;
using std::cout;
namespace sh = shogun;

namespace FT{
	
	/*!
     * @class ML
     * @brief class that specifies the machine learning algorithm to pair with Fewtwo. 
     */
    class ML 
    {
        public:
        	
            ML(string ml, bool classification)
            {
                /*!
                 * use string to specify a desired ML algorithm from shogun.
                 */
                
                type = ml;
                
                auto prob_type = sh::EProblemType::PT_REGRESSION;
                
                if (classification)
                     prob_type = sh::EProblemType::PT_MULTICLASS;               

                if (!ml.compare("LeastAngleRegression"))
                    p_est = make_shared<sh::CLeastAngleRegression>();
                
                else if (!ml.compare("RandomForest")){
                    p_est = make_shared<sh::CRandomForest>();
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->
                                                               set_machine_problem_type(prob_type);
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_bags(100);
                                       
                    if (classification)
                    {
                        auto CR = some<sh::CMajorityVote>();                        
                        dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_combination_rule(CR);
                    }
                    else
                    {
                        auto CR = some<sh::CMeanRule>();
                        dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_combination_rule(CR);
                    }
                    
                }
                else if (!ml.compare("CART")){
                    p_est = make_shared<sh::CMyCARTree>();
                    dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                               set_machine_problem_type(prob_type);
                    dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                               set_max_depth(4);                
                }

                else if (!ml.compare("LinearRidgeRegression"))
                    p_est = make_shared<sh::CLinearRidgeRegression>();
                    
                else if (!ml.compare("LinearLogisticRegression"))
                    p_est = make_shared<sh::CLibLinearRegression>();
                
                else if (!ml.compare("SVM"))
                {
                
                	if(classification)
                		p_est = make_shared<sh::CLibLinear>(sh::L2R_L2LOSS_SVC_DUAL);
	                    
	                else
	                	p_est = make_shared<sh::CLibLinearRegression>();
	            }
	            
	            else if (!ml.compare("LR"))
	            	p_est = make_shared<sh::CLibLinear>(sh::L2R_LR);
	            
	            else
                	std::cerr << "'" + ml + "' is not a valid ml choice\n";
                
            }
        
            ~ML(){}

            // return vector of weights for model. 
            vector<double> get_weights();
            
            // train ml model on X and return estimation y. 
            VectorXd out(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                         const vector<char>& dtypes=vector<char>());
            
            // set data types (for tree-based methods)            
            void set_dtypes(const vector<char>& dtypes)
            {
                assert (!type.compare("CART") || !type.compare("RandomForest"));

                // set attribute types True if boolean, False if continuous/ordinal
                sh::SGVector<bool> dt(dtypes.size());
                for (unsigned i = 0; i< dtypes.size(); ++i)
                    dt[i] = dtypes[i] == 'b';
                if (!type.compare("CART"))
                    dynamic_pointer_cast<sh::CMyCARTree>(p_est)->set_feature_types(dt);
                else if (!type.compare("RandomForest"))
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_feature_types(dt);
            }
            shared_ptr<sh::CMachine> p_est;
            string type;
    };
/////////////////////////////////////////////////////////////////////////////////////// Definitions

    vector<double> ML::get_weights()
    {    
        /*!
         * return weight vector from model.
         */
        vector<double> w;
        
        if (!type.compare("LeastAngleRegression") || !type.compare("LinearRidgeRegression")||
        	!type.compare("SVM") || (!type.compare("LR")))
        {
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
                
        }
        else if (!type.compare("CART"))
        {
            w = dynamic_pointer_cast<sh::CMyCARTree>(p_est)->feature_importances();
        }

        return softmax(w);
    }

    VectorXd ML::out(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                     const vector<char>& dtypes)
    { 
    	/*!
         * Trains ml on X, y to generate output yhat = f(X). 
         *
         *  Input: 
         
         *       X: n_features x n_samples matrix
         *       y: n_samples vector of training labels
         *       params: fewtwo parameters
         *       ml: the ML model to be trained on X
         
         *  Output:
         
         *       yhat: n_samples vector of outputs
        */ 
        
        // for random forest we need to set the number of features per bag

        if (!type.compare("RandomForest"))
        {
            //std::cout << "setting max_feates\n";
            // set max features to sqrt(n_features)
            int max_feats = std::sqrt(X.rows());
            dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_random_features(max_feats);
        }
        // for tree-based methods we need to specify data types 
        if (!type.compare("RandomForest") || !type.compare("CART"))
        {            
            //std::cout << "setting dtypes\n";
            if (dtypes.empty())
                set_dtypes(params.dtypes);
            else
                set_dtypes(dtypes);
        }
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " normalize features\n";
        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i){
            if (std::isinf(X.row(i).norm()))
            {
                X.row(i) = VectorXd::Zero(X.row(i).size());
                continue;
            }
            X.row(i) = X.row(i).array() - X.row(i).mean();
            if (X.row(i).norm() > NEAR_ZERO)
                X.row(i).normalize();
        }
        //X.rowwise().normalize();
                // define shogun data
        //if (params.verbosity > 1) 
        //    std::cout << "thread " + std::to_string(omp_get_thread_num()) + " X: " << X << "\n"; 

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        
        if((!params.ml.compare("SVM") && params.classification) || !params.ml.compare("LR"))           	
        	ml->p_est->set_labels(some<CBinaryLabels>(SGVector<float64_t>(y), 0.5));       	
        else if (params.classification)       
            p_est->set_labels(some<CMulticlassLabels>(SGVector<float64_t>(y)));
        else
            p_est->set_labels(some<CRegressionLabels>(SGVector<float64_t>(y)));
        //std::cout << "past set labels\n"; 

        // train ml
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " train\n";
        params.msg("ML training on thread" + std::to_string(omp_get_thread_num()) + "...",2," ");
        //#pragma omp critical
        {
            p_est->train(features);
        }
        params.msg("done.",2);
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " get output\n";
        //get output
        SGVector<double> y_pred; 

        if (params.classification)
        {
            auto clf = p_est->apply_multiclass(features);
            y_pred = clf->get_labels();
            delete clf;
            
        }
        else
        {
            auto reg = p_est->apply_regression(features);
            y_pred = reg->get_labels();
            delete reg;
        }
        //y_pred.display_vector();
        // map to Eigen vector
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        //std::cout << "weights\n";
        // weights
        vector<double> w = get_weights();

        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " map to vector\n";
        
        if (Eigen::isinf(yhat.array()).any() || Eigen::isnan(yhat.array()).any())
        {
            std::cerr << "inf or nan values in model fit to: " << X << "\n";
            pass = false;
        }
        //std::cout << "yhat is " << yhat.transpose() << std::endl; 
        // return
        return yhat;
    }

}


#endif
