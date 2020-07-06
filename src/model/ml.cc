/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "ml.h"                                                    

using namespace shogun;

namespace FT{ 
namespace Model{
    
// global default ML parameters
map<ML_TYPE, float> C_DEFAULT = {
    {LARS, 0},
    {Ridge, 1e-6},
    {LR, 1.0},
    {L1_LR, 1.0}
};

ML::ML(string ml, bool norm, bool classification, int n_classes)
{
    /*!
     * use string to specify a desired ML algorithm from shogun.
     */
    ml_hash["Lasso"] = LARS;         
    ml_hash["LinearRidgeRegression"] = Ridge;
    ml_hash["Ridge"] = Ridge;
    ml_hash["SVM"] = SVM;
    ml_hash["RandomForest"] = RF;
    ml_hash["CART"] = CART;
    ml_hash["LR"] = LR;
    ml_hash["L2_LR"] = LR;
    ml_hash["L1_LR"] = L1_LR;
    ml_hash["RF"] = RF;

    ml_str  = ml;

    if ( ml_hash.find(ml) == ml_hash.end() ) 
    {
        // not found
        HANDLE_ERROR_THROW("ml type '" + ml + "' not defined");
    } 
    else 
        this->ml_type = ml_hash.at(ml);
    
    this->prob_type = PT_REGRESSION;
    max_train_time=30; 
    normalize = norm;
    if (classification)
    { 
        if (n_classes==2)
            this->prob_type = PT_BINARY;
        else
            this->prob_type = PT_MULTICLASS;               
    }
    this->C = C_DEFAULT.at(ml_type);
}

void ML::init()
{
    // set up ML based on type
    if (ml_type == LARS)
    {
        p_est = make_shared<sh::CLeastAngleRegression>(true);
        dynamic_pointer_cast<sh::CLeastAngleRegression>(
                p_est)->set_max_non_zero(int(this->C));
    }
    else if (ml_type == Ridge)
    {
        p_est = make_shared<sh::CLinearRidgeRegression>();
        dynamic_pointer_cast<sh::CLinearRidgeRegression>(
                p_est)->set_compute_bias(true);
        dynamic_pointer_cast<sh::CLinearRidgeRegression>(
                p_est)->set_tau(this->C);
    }
    else if (ml_type == RF)
    {
        p_est = make_shared<sh::CMyRandomForest>();
        dynamic_pointer_cast<sh::CMyRandomForest>(
                p_est)->set_machine_problem_type(this->prob_type);
        dynamic_pointer_cast<sh::CMyRandomForest>(
                p_est)->set_num_bags(10);
                           
        if (this->prob_type != PT_REGRESSION)
        {
            auto CR = some<sh::CMajorityVote>();                        
            dynamic_pointer_cast<sh::CMyRandomForest>(
                    p_est)->set_combination_rule(CR);
        }
        else
        {
            auto CR = some<sh::CMeanRule>();
            dynamic_pointer_cast<sh::CMyRandomForest>(
                    p_est)->set_combination_rule(CR);
        }
        
    }
    else if (ml_type == CART)
    {
        p_est = make_shared<sh::CMyCARTree>();
        dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->set_machine_problem_type(this->prob_type);
        dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->set_max_depth(6);                
    }
                   
    else if (ml_type == SVM)
    {               
        if(this->prob_type==PT_BINARY)
            p_est = make_shared<sh::CMyLibLinear>(
                    sh::L2R_L2LOSS_SVC_DUAL);       
        else if (this->prob_type==PT_MULTICLASS){
            p_est = make_shared<CMyMulticlassLibLinear>();
            dynamic_pointer_cast<CMyMulticlassLibLinear>(
                    p_est)->set_prob_heuris(sh::OVA_NORM);

        }
        else                // SVR
            p_est = make_shared<sh::CLibLinearRegression>(); 
        
    }
    else if (in({LR, L1_LR}, ml_type))
    {
        assert(this->prob_type!=PT_REGRESSION 
                && "LR only works with classification.");
        if (this->prob_type == PT_BINARY){
            if (this->ml_type == LR)
            {
                p_est = make_shared<sh::CMyLibLinear>(
                    sh::L2R_LR);
            }
            else
            {
                p_est = make_shared<sh::CMyLibLinear>(sh::L1R_LR);
            }

            // setting parameters to match sklearn defaults
            dynamic_pointer_cast<sh::CMyLibLinear>(
                        p_est)->set_bias_enabled(true);
            dynamic_pointer_cast<sh::CMyLibLinear>(
                    p_est)->set_epsilon(0.0001);
            dynamic_pointer_cast<sh::CMyLibLinear>(
                    p_est)->set_max_iterations(100);
            dynamic_pointer_cast<sh::CMyLibLinear>(
                    p_est)->set_C(this->C,this->C); 
            //cout << "set ml type to CMyLibLinear\n";
        }
        else    // multiclass  
        {
            p_est = make_shared<sh::CMulticlassLogisticRegression>();
            dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(
                    p_est)->set_prob_heuris(sh::OVA_NORM);
            dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(
                    p_est)->set_z(this->C);
            dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(
                    p_est)->set_epsilon(0.0001);
            dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(
                    p_est)->set_max_iter(100);

        }

    
    }
    else
        HANDLE_ERROR_NO_THROW("'" + ml_str 
                + "' is not a valid ml choice\n");
    // set maximum training time per model 
    p_est->set_max_train_time(max_train_time);          
}

ML::~ML(){}

void ML::set_dtypes(const vector<char>& dtypes)
{
    if (ml_type == CART || ml_type == RF)
    {
        // set attribute types True if boolean, False if 
        // continuous/ordinal
        sh::SGVector<bool> dt(dtypes.size());
        for (unsigned i = 0; i< dtypes.size(); ++i)
            dt[i] = dtypes.at(i) == 'b';
        if (ml_type == CART)
            dynamic_pointer_cast<sh::CMyCARTree>(
                    p_est)->set_feature_types(dt);
        else if (ml_type == RF)
            dynamic_pointer_cast<sh::CMyRandomForest>(
                    p_est)->set_feature_types(dt);
    }
    this->dtypes = dtypes;
}

vector<float> ML::get_weights()
{    
    /*!
     * @return weight vector from model.
     */
    vector<double> w;
    
    if (in({LARS, Ridge, SVM, LR, L1_LR}, ml_type))
    {
        // for multiclass, return the average weight magnitude over 
        // the OVR models
        if(this->prob_type == PT_MULTICLASS 
                && in({LR, L1_LR, SVM}, ml_type) ) 
        {
            /* cout << "in get_weights(), multiclass LR\n"; */
            vector<SGVector<double>> weights;

            if( in({LR, L1_LR}, ml_type))
                weights = dynamic_pointer_cast<
                    sh::CMulticlassLogisticRegression>(p_est)->get_w();
            else //SVM
                weights = \
                  dynamic_pointer_cast<sh::CMyMulticlassLibLinear>(
                          p_est)->get_w();
       
            /* cout << "set weights from get_w()\n"; */
                
            /* for( int j = 0;j<weights.at(0).size(); j++) */ 
            /*     w.push_back(0); */
            w = vector<double>(weights.at(0).size());
            /* cout << "weights.size(): " << weights.size() << "\n"; */
            /* cout << "w size: " << w.size() << "\n"; */
            /* cout << "getting abs weights\n"; */
            
            for( int i = 0 ; i < weights.size(); ++i )
            {
                /* cout << "weights:\n"; */
                /* weights.at(i).display_vector(); */

                for( int j = 0;j<weights.at(i).size(); ++j) 
                {
                    w.at(j) += fabs(weights.at(i)[j]);
                    /* w.at(j) += weights.at(i)[j]; */
                }
            }
            /* cout << "normalizing weights\n"; */ 
            for( int i = 0; i < w.size() ; i++) 
                w.at(i) = w.at(i)/weights.size(); 
            
            return vector<float>(w.begin(), w.end());
        }
        else
        {
            // otherwise, return the true weights 
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(
                    p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
        }
    }
    else if (ml_type == CART)           
        w = dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->feature_importances();
    else
        w = dynamic_pointer_cast<sh::CMyRandomForest>(
                p_est)->feature_importances();

    return vector<float>(w.begin(), w.end());
}

shared_ptr<CLabels> ML::fit(MatrixXf& X, VectorXf& y, 
        const Parameters& params, bool& pass,
                 const vector<char>& dtypes)
{ 
    /*!
     * Trains ml on X, y to generate output yhat = f(X). 
     *     
     * @param X: n_features x n_samples matrix
     * @param  y: n_samples vector of training labels
     * @param params: feat parameters
     * @param[out] pass: returns True if fit was successful, 
     * False if not
     * @param dtypes: the data types of features in X
     *
     * @return yhat: n_samples vector of outputs
    */ 
    
    // for random forest we need to set the number of features per bag
    init();

    if (ml_type == RF)
    {
        int max_feats = std::sqrt(X.rows());
        dynamic_pointer_cast<sh::CMyRandomForest>(
                p_est)->set_num_random_features(max_feats);
    }
    
    // for tree-based methods we need to specify data types 
    if (ml_type == RF || ml_type == CART)
    {            
        //std::cout << "setting dtypes\n";
        if (dtypes.empty())
            set_dtypes(params.dtypes);
        else
            set_dtypes(dtypes);
    }

    if (normalize)
    {
        /* N.fit_normalize(X, find_dtypes(X)); */
        if (dtypes.empty())
            N.fit_normalize(X, find_dtypes(X));  
        else 
            N.fit_normalize(X, dtypes);
    }

    /* else */
        /* cout << "normlize is false\n"; */
        
    MatrixXd _X = X.template cast<double>();
    VectorXd _y = y.template cast<double>();
    
    
    if(_X.isZero(0.0001))
    {

        logger.log("Setting labels to zero since features are zero\n", 
                2);

        shared_ptr<CLabels> labels;
        
        switch (this->prob_type)
        {
            case PT_REGRESSION : 
                labels = std::shared_ptr<CLabels>(
                        new CRegressionLabels(_y.size()));
                break;
            case PT_BINARY     : 
                labels = std::shared_ptr<CLabels>(
                        new CBinaryLabels(_y.size()));
                break;
            case PT_MULTICLASS : 
                labels = std::shared_ptr<CLabels>(
                        new CMulticlassLabels(_y.size()));
                break;
        }

        pass = false;
        return labels;
    }
    

    auto features = some<CDenseFeatures<float64_t>>(
            SGMatrix<float64_t>(_X));
    
    // for liblinear L1, we have to transpose the features during training
    if (ml_type == L1_LR && this->prob_type==PT_BINARY)
        features = features->get_transposed();


     
    if(this->prob_type==PT_BINARY && in({LR, L1_LR, SVM}, ml_type))
    { 
        // binary classification
        p_est->set_labels(
            some<CBinaryLabels>(SGVector<float64_t>(_y), 0.5));       	
    }
    else if (this->prob_type==PT_MULTICLASS)                         
        // multiclass classification       
        p_est->set_labels(some<CMulticlassLabels>(
                    SGVector<float64_t>(_y)));
    else                                                    
        // regression
        p_est->set_labels(some<CRegressionLabels>(
                    SGVector<float64_t>(_y)));
    
    // train ml
    logger.log("ML training on thread " 
               + std::to_string(omp_get_thread_num()) + "...",3," ");
    // *** Train the model ***  
    p_est->train(features);

    logger.log("done!",3);
   
    // tranpose features back
    if (ml_type == L1_LR && this->prob_type==PT_BINARY)
        features = features->get_transposed();

    logger.log("exiting ml::fit",3); 
    return this->retrieve_labels(features, true, pass); 
}

VectorXf ML::fit_vector(MatrixXf& X, VectorXf& y, 
        const Parameters& params, bool& pass,
                 const vector<char>& dtypes)
{
    shared_ptr<CLabels> labels = fit(X, y, params, pass, dtypes); 
    
    return labels_to_vector(labels);     
}


shared_ptr<CLabels> ML::predict(MatrixXf& X, bool print)
{
    logger.log("ML::predict...",3);
    shared_ptr<CLabels> labels;
    // make sure the model fit() method passed
    if (get_weights().empty())
    {
        HANDLE_ERROR_NO_THROW("weight empty; returning zeros"); 
        if (this->prob_type==PT_BINARY) 
        {
            labels = std::shared_ptr<CLabels>(
                    new CBinaryLabels(X.cols()));
            for (unsigned i = 0; i < X.cols() ; ++i)
            {
                dynamic_pointer_cast<CBinaryLabels>(
                        labels)->set_value(0,i);
                dynamic_pointer_cast<CBinaryLabels>(
                        labels)->set_label(i,0);
            }
            return labels;
        }
        else if (this->prob_type == PT_MULTICLASS)
        {
            labels = std::shared_ptr<CLabels>(
                    new CMulticlassLabels(X.cols()));
            for (unsigned i = 0; i < X.cols() ; ++i)
            {
                dynamic_pointer_cast<CMulticlassLabels>(
                        labels)->set_value(0,i);
                dynamic_pointer_cast<CMulticlassLabels>(
                        labels)->set_label(i,0);
            }
            return labels;
        }
        else
        {
            labels = std::shared_ptr<CLabels>(
                    new CRegressionLabels(X.cols()));
            for (unsigned i = 0; i < X.cols() ; ++i)
            {
                dynamic_pointer_cast<CRegressionLabels>(
                        labels)->set_value(0,i);
                dynamic_pointer_cast<CRegressionLabels>(
                        labels)->set_label(i,0);
            }
            return labels;
        }
    }
    /* cout << "weights: \n"; */
    /* for (const auto& w : get_weights()) */
    /*     cout << w << ", " ; */
    /* cout << "\n"; */
    /* cout << "normalize\n"; */ 
    if (normalize)
        N.normalize(X);
    
    MatrixXd _X = X.template cast<double>();
    auto features = some<CDenseFeatures<float64_t>>(
            SGMatrix<float64_t>(_X));
     
    bool pass = true; 
    return this->retrieve_labels(features, true, pass);
}

VectorXf ML::predict_vector(MatrixXf& X)
{
    shared_ptr<CLabels> labels = predict(X);
    return labels_to_vector(labels);     
    
}

ArrayXXf ML::predict_proba(MatrixXf& X)
{
    shared_ptr<CLabels> labels = shared_ptr<CLabels>(predict(X));
       
    if (this->prob_type==PT_BINARY 
            && in({SVM, LR, L1_LR, CART, RF}, ml_type)) 
    {
        shared_ptr<CBinaryLabels> BLabels = \
                        dynamic_pointer_cast<CBinaryLabels>(labels);
        /* BLabels->scores_to_probabilities(); */
        SGVector<double> tmp= BLabels->get_values();
        ArrayXXd confidences(1,tmp.size());
        confidences.row(0) = Map<ArrayXd>(tmp.data(),tmp.size()); 
        return confidences.template cast<float>();
    }
    else if (this->prob_type == PT_MULTICLASS)
    {
        shared_ptr<CMulticlassLabels> MLabels = \
                    dynamic_pointer_cast<CMulticlassLabels>(labels);
        MatrixXd confidences(MLabels->get_num_classes(), 
                MLabels->get_num_labels()) ; 
        for (unsigned i =0; i<confidences.rows(); ++i)
        {
            SGVector<double> tmp = \
                           MLabels->get_multiclass_confidences(int(i));
            confidences.row(i) = Map<ArrayXd>(tmp.data(),tmp.size());
            /* std::cout << confidences.row(i) << "\n"; */
        }
        return confidences.template cast<float>();;
    }
    else
        HANDLE_ERROR_THROW("Error: predict_proba not defined for "
                "problem type or ML method");
}

VectorXf ML::labels_to_vector(const shared_ptr<CLabels>& labels)
{
    SGVector<double> y_pred;
    if (this->prob_type==PT_BINARY 
            && in({SVM, LR, L1_LR, CART, RF}, ml_type)) 
        y_pred = dynamic_pointer_cast<sh::CBinaryLabels>(
                labels)->get_labels();
    else if (this->prob_type != PT_REGRESSION)
        y_pred = dynamic_pointer_cast<sh::CMulticlassLabels>(
                labels)->get_labels();
    else
        y_pred = dynamic_pointer_cast<sh::CRegressionLabels>(
                labels)->get_labels();
   
    Map<VectorXd> yhat(y_pred.data(),y_pred.size());
    
    if (this->prob_type==PT_BINARY 
            && in({SVM, LR, L1_LR, CART, RF}, ml_type)) 
        // convert -1 to 0
        yhat = (yhat.cast<int>().array() == -1).select(0,yhat);

    VectorXf yhatf = yhat.template cast<float>();
    clean(yhatf);
    return yhatf;
}
float ML::get_bias()
{   // get bias weight. only works with linear machines
    if (in({L1_LR, LR, LARS, Ridge}, ml_type) )
    {
        return dynamic_pointer_cast<sh::CLinearMachine>(
                p_est)->get_bias();
    }
    else
        return 0;

}
shared_ptr<CLabels> ML::retrieve_labels(CDenseFeatures<float64_t>* features, 
                                   bool proba, bool& pass)
{
    logger.log("ML::get_labels",3);
    shared_ptr<CLabels> labels;
    SGVector<double> y_pred; 


    if (this->prob_type==PT_BINARY && 
            in({LR, L1_LR, SVM, CART, RF}, ml_type))
    {
        labels = shared_ptr<CLabels>(
                p_est->apply_binary(features));

        if (proba)
        {
            if (ml_type == CART)
            {
                dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                    set_probabilities(labels.get(), features);
            }
            else if(ml_type == RF)
            {
                dynamic_pointer_cast<sh::CMyRandomForest>(p_est)->
                    set_probabilities(labels.get(), features);
            }
            else
            {
                    dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->
                        set_probabilities(labels.get(), features);
            }
        }
        y_pred = dynamic_pointer_cast<sh::CBinaryLabels>(
                labels)->get_labels();
       
    }
    else if (this->prob_type != PT_REGRESSION)                         
    // multiclass classification
    {
        labels = shared_ptr<CLabels>(
                p_est->apply_multiclass(features));
        y_pred = dynamic_pointer_cast<sh::CMulticlassLabels>(
                labels)->get_labels();
    }
    else                                                    
    // regression
    {
        labels = shared_ptr<CLabels>(
                p_est->apply_regression(features));
        y_pred = dynamic_pointer_cast<sh::CRegressionLabels>(
                labels)->get_labels();
    }
    // map to Eigen vector
    Map<VectorXd> yhat(y_pred.data(),y_pred.size());
   

    if (isinf(yhat.array()).any() || isnan(yhat.array()).any() 
            || yhat.size()==0)
        pass = false;

    return labels;
}
shared_ptr<CLabels> ML::fit_tune(MatrixXf& X, VectorXf& y, 
        const Parameters& params, bool& pass, const vector<char>& dtypes, 
        bool set_default)
{
    logger.log("tuning C...",2);
    LongData Z;
    DataRef d_cv(X, y, Z, params.classification, 
            params.protected_groups);
    FT::Eval::Scorer S(params.scorer);
    // for linear models, tune the regularization strength
    if (in({LARS, Ridge, L1_LR, LR}, this->ml_type) )
    {
        vector<float> Cs;
        switch (this->ml_type)
        {
            case LARS:
                // in this case C is the max num of non-zero variables
                Cs.resize(X.rows()-1);
                iota(Cs.begin(),Cs.end(),1);
                break;
            case Ridge:
                Cs = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2};
                break;
            default:
                Cs = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3};
        }

        float n_splits = 10;
        MatrixXf losses(Cs.size(),int(n_splits));
        VectorXf dummy;

        for (int i = 0; i < n_splits; ++i)
        {
            logger.log("split " + to_string(i) + "...",3);
            d_cv.train_test_split(true, 0.8);

            for (int j = 0; j< Cs.size(); ++j)
            {
                this->C = Cs.at(j);
                this->fit(d_cv.t->X, d_cv.t->y, 
                        params, pass, this->dtypes);

                losses(j,i) = S.score(d_cv.v->y, 
                                    this->predict(d_cv.v->X), 
                                    dummy, params.class_weights);
            }
        }
        // get mean loss for each C
        VectorXf mean_loss = losses.rowwise().mean();
        string cv_report = "mean_loss (" + to_string(mean_loss.size()) 
            + "): \n" ;
        for (int i = 0; i < Cs.size(); ++i)
            cv_report += "C = " + to_string(Cs.at(i)) + ", mean_loss = " 
                + to_string(mean_loss(i)) + "\n";
        VectorXf::Index min_index;
        float min_loss = mean_loss.minCoeff(&min_index);
        float best_C = Cs.at(min_index);
        cv_report += "best C: " + to_string(best_C) + "\n" ;
        logger.log(cv_report, 2);
        // set best C and fit a final model to all data with it
        this->C = best_C;
        if (set_default)
        {
            C_DEFAULT.at(this->ml_type) = best_C;
            logger.log("changing C_DEFAULT: " 
                    + to_string(C_DEFAULT[ml_type]), 2);
        }
        return this->fit(X, y, params, pass, dtypes);
    }
}


} // namespace Model
} // namespace FT
