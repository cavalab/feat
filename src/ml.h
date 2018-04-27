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
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <cmath>
// internal includes
#include "ml/MyCARTree.h"
#include "ml/MulticlassLogisticRegression.h"

// stuff being used
using std::string;
using std::dynamic_pointer_cast;
using std::cout;
namespace sh = shogun;
using sh::EProblemType; 
namespace FT{
	
	/*!
     * @class ML
     * @brief class that specifies the machine learning algorithm to pair with Feat. 
     */
    
    class ML 
    {
        public:
        	
            ML(const Parameters& params)
            {
                /*!
                 * use string to specify a desired ML algorithm from shogun.
                 */
                
                ml_type = params.ml;
                prob_type = PT_REGRESSION;
                
                if (params.classification)
                { 
                    if (params.n_classes==2)
                        prob_type = PT_BINARY;
                    else
                        prob_type = PT_MULTICLASS;               
                }
            }
            
            void init()
            {
                // set up ML based on type
                if (!ml_type.compare("LeastAngleRegression"))
                    p_est = make_shared<sh::CLeastAngleRegression>();
                else if (!ml_type.compare("LinearRidgeRegression"))
                    p_est = make_shared<sh::CLinearRidgeRegression>();
                else if (!ml_type.compare("RandomForest"))
                {
                    p_est = make_shared<sh::CRandomForest>();
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->
                                                               set_machine_problem_type(prob_type);
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_bags(100);
                                       
                    if (prob_type != PT_REGRESSION)
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
                else if (!ml_type.compare("CART"))
                {
                    p_est = make_shared<sh::CMyCARTree>();
                    dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                               set_machine_problem_type(prob_type);
                    dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                               set_max_depth(6);                
                }
                               
                else if (!ml_type.compare("SVM"))
                {               
                	if(prob_type==PT_BINARY)
                        p_est = make_shared<sh::CLibLinear>(sh::L2R_L2LOSS_SVC_DUAL);       
                    else if (prob_type==PT_MULTICLASS)
                        p_est = make_shared<sh::CMulticlassLibLinear>();
	                else                // SVR
	                	p_est = make_shared<sh::CLibLinearRegression>(); 
                    
	            }
	            else if (!ml_type.compare("LR"))
                {
                    assert(prob_type!=PT_REGRESSION && "LR only works with classification.");
                    if (prob_type == PT_BINARY){
	            	    p_est = make_shared<sh::CLibLinear>(sh::L2R_LR);
                        // setting parameters to match sklearn defaults
                        dynamic_pointer_cast<sh::CLibLinear>(p_est)->set_compute_bias(true);
                        dynamic_pointer_cast<sh::CLibLinear>(p_est)->set_epsilon(0.0001);
                        //cout << "set ml type to CLibLinear\n";
                    }
                    else    // multiclass 
                        p_est = make_shared<sh::CMulticlassLogisticRegression>();
                
                }
	            else
                	std::cerr << "'" + ml_type + "' is not a valid ml choice\n";
                
            }
        
            ~ML(){}

            // return vector of weights for model. 
            vector<double> get_weights();
            
            // train ml model on X and return estimation y. 
            VectorXd fit(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                         const vector<char>& dtypes=vector<char>());
            // predict using a trained ML model, returning an estimation. 
            VectorXd predict(MatrixXd& X);
            // set data types (for tree-based methods)            
            void set_dtypes(const vector<char>& dtypes)
            {
                if (!ml_type.compare("CART") || !ml_type.compare("RandomForest"))
                {
                    // set attribute types True if boolean, False if continuous/ordinal
                    sh::SGVector<bool> dt(dtypes.size());
                    for (unsigned i = 0; i< dtypes.size(); ++i)
                        dt[i] = dtypes[i] == 'b';
                    if (!ml_type.compare("CART"))
                        dynamic_pointer_cast<sh::CMyCARTree>(p_est)->set_feature_types(dt);
                    else if (!ml_type.compare("RandomForest"))
                        dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_feature_types(dt);
                }
            }

            shared_ptr<sh::CMachine> p_est;     ///< pointer to the ML object
            string ml_type;                        ///< user specified ML type
            sh::EProblemType prob_type;         ///< type of learning problem; binary, multiclass 
                                                ///  or regression 
            Normalizer N;                       // normalization
    };
/////////////////////////////////////////////////////////////////////////////////////// Definitions

    vector<double> ML::get_weights()
    {    
        /*!
         * return weight vector from model.
         */
        vector<double> w;
        
        if (!ml_type.compare("LeastAngleRegression") || !ml_type.compare("LinearRidgeRegression")||
        	!ml_type.compare("SVM") || (!ml_type.compare("LR")))
        {
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
            for (unsigned i =0; i<w.size(); ++i)    // take absolute value of weights
                w[i] = fabs(w[i]);
        }
        else if (!ml_type.compare("CART"))           
            w = dynamic_pointer_cast<sh::CMyCARTree>(p_est)->feature_importances();
        else
        {
            std::cerr << "ERROR: ML::get_weights not implemented for " + ml_type << "\n";
            
        }
        
        return w;
    }

    VectorXd ML::fit(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                     const vector<char>& dtypes)
    { 
    	/*!
         * Trains ml on X, y to generate output yhat = f(X). 
         *
         *  Input: 
         
         *       X: n_features x n_samples matrix
         *       y: n_samples vector of training labels
         *       params: feat parameters
         *       ml: the ML model to be trained on X
         
         *  Output:
         
         *       yhat: n_samples vector of outputs
        */ 
        
                // for random forest we need to set the number of features per bag

        init();
        if (!ml_type.compare("RandomForest"))
        {
            //std::cout << "setting max_feates\n";
            // set max features to sqrt(n_features)
            int max_feats = std::sqrt(X.rows());
            dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_random_features(max_feats);
        }
        // for tree-based methods we need to specify data types 
        if (!ml_type.compare("RandomForest") || !ml_type.compare("CART"))
        {            
            //std::cout << "setting dtypes\n";
            if (dtypes.empty())
                set_dtypes(params.dtypes);
            else
                set_dtypes(dtypes);
        }
        
        if (dtypes.empty())
            N.fit_normalize(X, params.dtypes);  
        else 
            N.fit_normalize(X, dtypes);
         
        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        //std::cout << "setting labels (n_classes = " << params.n_classes << ")\n"; 
        //cout << "y is " << y.transpose() << "\n";
        if(prob_type==PT_BINARY && 
                (!ml_type.compare("LR") || !ml_type.compare("SVM")))  // binary classification           	
        	p_est->set_labels(some<CBinaryLabels>(SGVector<float64_t>(y), 0.5));       	
        else if (prob_type!=PT_REGRESSION)                         // multiclass classification       
            p_est->set_labels(some<CMulticlassLabels>(SGVector<float64_t>(y)));
        else                                                    // regression
            p_est->set_labels(some<CRegressionLabels>(SGVector<float64_t>(y)));
        
        // train ml
        params.msg("ML training on thread" + std::to_string(omp_get_thread_num()) + "...",2," ");       

        // *** Train the model ***  
        p_est->train(features);
        // *** Train the model ***
        
        params.msg("done.",2);
       
        //get output
        SGVector<double> y_pred; 

        if (prob_type==PT_BINARY && 
             (!ml_type.compare("LR") || !ml_type.compare("SVM")))     // binary classification
        {
            auto clf = p_est->apply_binary(features);
            /* clf->scores_to_probabilities(0.0,0.0);  // get sigmoid-fn probabilities */
            /* y_pred = clf->get_values(); */
            y_pred = clf->get_labels();
            delete clf;
        }
        else if (params.classification)                         // multiclass classification
        {
            auto clf = p_est->apply_multiclass(features);
            y_pred = clf->get_labels();
            delete clf;
            
        }
        else                                                    // regression
        {
            auto reg = p_est->apply_regression(features);
            y_pred = reg->get_labels();
            delete reg;
        }
        //y_pred.display_vector();
        // map to Eigen vector
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " map to vector\n";
        
        if (isinf(yhat.array()).any() || isnan(yhat.array()).any())
        {
            std::cerr << "inf or nan values in model fit to: " << X << "\n";
            pass = false;
        }
        /* std::cout << "ML::fit yhat is " << yhat.transpose() << std::endl; */ 
        // return
        return yhat;
    }

    VectorXd ML::predict(MatrixXd& X)
    {

        N.normalize(X);
        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        
        SGVector<double> y_pred;
        if (prob_type==PT_BINARY && 
                (!ml_type.compare("SVM") || !ml_type.compare("LR")))
        {
            auto tmp = p_est->apply_binary(features);
            y_pred = tmp->get_labels();
            delete tmp;
        }
        else if (prob_type != PT_REGRESSION)
        {
            auto tmp = p_est->apply_multiclass(features);
            y_pred = tmp->get_labels();
            delete tmp;
        }
        else
        {
            auto tmp = p_est->apply_regression(features);
            y_pred = tmp->get_labels();
            delete tmp;
        }
        
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        
        if (prob_type==PT_BINARY && (!ml_type.compare("LR") || !ml_type.compare("SVM")))
            // convert -1 to 0
            yhat = (yhat.cast<int>().array() == -1).select(0,yhat);
        
        /* std::cout << "ML::predict yhat:" <<  yhat.transpose() << "\n"; */
        return yhat ;
    }

}


#endif
