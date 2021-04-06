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
        THROW_INVALID_ARGUMENT("ml type '" + ml + "' not defined");
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
    this->init(true);
}

void ML::init(bool assign_p_est)
{
    if (assign_p_est)
        this->p_est.reset();
    // set up ML based on type
    if (ml_type == LARS)
    {
        if (assign_p_est)
            p_est = make_shared<sh::CLeastAngleRegression>(true);
        dynamic_pointer_cast<sh::CLeastAngleRegression>(
                p_est)->set_max_non_zero(int(this->C));
    }
    else if (ml_type == Ridge)
    {
        if (assign_p_est)
            p_est = make_shared<sh::CLinearRidgeRegression>();
        // auto typed_p_est = typed<sh::CLinearRidgeRegression>();
        auto typed_p_est = dynamic_pointer_cast<sh::CLinearRidgeRegression>(
                            p_est);
        typed_p_est->set_compute_bias(true);
        typed_p_est->set_tau(this->C);
    }
    else if (ml_type == RF)
    {
        if (assign_p_est)
            p_est = make_shared<sh::CMyRandomForest>();
        auto typed_p_est = dynamic_pointer_cast<sh::CMyRandomForest>(p_est);
        typed_p_est->set_machine_problem_type(this->prob_type);
        typed_p_est->set_num_bags(10);
                           
        if (this->prob_type != PT_REGRESSION)
        {
            auto CR = some<sh::CMajorityVote>();                        
            typed_p_est->set_combination_rule(CR);
        }
        else
        {
            auto CR = some<sh::CMeanRule>();
            typed_p_est->set_combination_rule(CR);
        }
        
    }
    else if (ml_type == CART)
    {
        if (assign_p_est)
            p_est = make_shared<sh::CMyCARTree>();
        dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->set_machine_problem_type(this->prob_type);
        dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->set_max_depth(6);                
    }
                   
    else if (ml_type == SVM)
    {               
        if(this->prob_type==PT_BINARY)
            if (assign_p_est)
                p_est = make_shared<sh::CMyLibLinear>(
                        sh::L2R_L2LOSS_SVC_DUAL);       
        else if (this->prob_type==PT_MULTICLASS){
            if (assign_p_est)
                p_est = make_shared<CMyMulticlassLibLinear>();
                dynamic_pointer_cast<CMyMulticlassLibLinear>(
                        p_est)->set_prob_heuris(sh::OVA_SOFTMAX);

        }
        else                // SVR
            if (assign_p_est)
                p_est = make_shared<sh::CLibLinearRegression>(); 
        
    }
    else if (in({LR, L1_LR}, ml_type))
    {
        assert(this->prob_type!=PT_REGRESSION 
                && "LR only works with classification.");
        if (this->prob_type == PT_BINARY){
            if (this->ml_type == LR)
            {
                if (assign_p_est)
                    p_est = make_shared<sh::CMyLibLinear>(
                        sh::L2R_LR);
            }
            else
            {
                if (assign_p_est)
                    p_est = make_shared<sh::CMyLibLinear>(sh::L1R_LR);
            }

            auto typed_p_est = dynamic_pointer_cast<sh::CMyLibLinear>(p_est);
            // setting parameters to match sklearn defaults
            typed_p_est->set_bias_enabled(true);
            typed_p_est->set_epsilon(0.0001);
            typed_p_est->set_max_iterations(100);
            typed_p_est->set_C(this->C,this->C); 
        }
        else    // multiclass  
        {
            if (assign_p_est)
                p_est = make_shared<sh::CMulticlassLogisticRegression>();

            auto typed_p_est = \
                dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(p_est);
            typed_p_est->set_prob_heuris(sh::OVA_SOFTMAX);
            typed_p_est->set_z(this->C);
            typed_p_est->set_epsilon(0.0001);
            typed_p_est->set_max_iter(100);

        }

    
    }
    else
        THROW_INVALID_ARGUMENT("'" + ml_str + "' is not a valid ml choice\n");
    // set maximum training time per model 
    p_est->set_max_train_time(max_train_time);          
}

ML::~ML()
{
    this->p_est.reset();
}

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

vector<float> ML::get_weights() const
{    
    /*!
     * @return weight vector from model.
     */
    vector<double> w;
    
    if (in({LARS, Ridge, SVM, LR, L1_LR}, ml_type))
    {
        /* For multiclass, return the average weight magnitude over 
         * the OVR models. These weights are normalized. 
         */
        if(this->prob_type == PT_MULTICLASS 
                && in({LR, L1_LR, SVM}, ml_type) ) 
        {
            vector<SGVector<double>> weights;

            if( in({LR, L1_LR}, ml_type))
                weights = dynamic_pointer_cast<
                    sh::CMulticlassLogisticRegression>(p_est)->get_w();
            else //SVM
                weights = \
                  dynamic_pointer_cast<sh::CMyMulticlassLibLinear>(
                          p_est)->get_w();
       
            if (weights.empty())
                return vector<float>();

            w = vector<double>(weights.at(0).size());
            
            // we have to average the absolute weights across 
            // estimators in order to return one weight for each feature.
            for( int i = 0 ; i < weights.size(); ++i )
            {
                for( int j = 0;j<weights.at(i).size(); ++j) 
                {
                    w.at(j) += fabs(weights.at(i)[j]);
                }
            }
            // normalize by the number of classes
            for( int i = 0; i < w.size() ; i++) 
                w.at(i) = w.at(i)/weights.size(); 
            
            return vector<float>(w.begin(), w.end());
        }
        /* For linear regression and binary classification
         * models, return the true weights. */
        else
        {
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(
                    p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
        }
    }
    /* For decision trees, return the feature importance scores.  */
    else if (ml_type == CART)           
        w = dynamic_pointer_cast<sh::CMyCARTree>(
                p_est)->feature_importances();
    else
        w = dynamic_pointer_cast<sh::CMyRandomForest>(
                p_est)->feature_importances();

    return vector<float>(w.begin(), w.end());
}

shared_ptr<CLabels> ML::fit(const MatrixXf& X, const VectorXf& y, 
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
    
    init(true);

    MatrixXd _X = X.cast<double>();
    VectorXd _y = y.cast<double>();

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
            N.fit_normalize(_X, find_dtypes(X));  
        else 
            N.fit_normalize(_X, dtypes);
    }

    /* else */
        /* cout << "normlize is false\n"; */
        
    
    
    if(_X.isZero(0.0001))
    {

        logger.log("Setting labels to zero since features are zero\n", 
                3);

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
   
    // transpose features back
    if (ml_type == L1_LR && this->prob_type==PT_BINARY)
        features = features->get_transposed();

    logger.log("exiting ml::fit",3); 
    auto y_pred = this->retrieve_labels(features, true, pass);
    features->free_features();
    return y_pred; 
}

VectorXf ML::fit_vector(const MatrixXf& X, const VectorXf& y, 
        const Parameters& params, bool& pass,
                 const vector<char>& dtypes)
{
    shared_ptr<CLabels> labels = fit(X, y, params, pass, dtypes); 
    
    return labels_to_vector(labels);     
}


shared_ptr<CLabels> ML::predict(const MatrixXf& X, bool print)
{
    logger.log("ML::predict...",3);
    shared_ptr<CLabels> labels;
    logger.log("X size: " + to_string(X.rows()) + "x" + to_string(X.cols()),3);
    MatrixXd _X = X.template cast<double>();
    logger.log("cast X to double",3);

    /* Make sure the model fit() method passed by
     * looking for empty weights.
     * If the weights are empty, assign dummy labels. */
    if (get_weights().empty())
    {
        logger.log("weight empty; returning zeros",3); 
        if (this->prob_type==PT_BINARY) 
        {
            labels = std::shared_ptr<CLabels>(
                    new CBinaryLabels(_X.cols()));
            for (unsigned i = 0; i < _X.cols() ; ++i)
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
                    new CMulticlassLabels(_X.cols()));
            for (unsigned i = 0; i < _X.cols() ; ++i)
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
                    new CRegressionLabels(_X.cols()));
            for (unsigned i = 0; i < _X.cols() ; ++i)
            {
                dynamic_pointer_cast<CRegressionLabels>(
                        labels)->set_value(0,i);
                dynamic_pointer_cast<CRegressionLabels>(
                        labels)->set_label(i,0);
            }
            return labels;
        }
    }

    /* Otherwise, apply normalization and retrieve labels
     * from the model. */
    if (normalize)
        N.normalize(_X);
    
    auto features = some<CDenseFeatures<float64_t>>(
            SGMatrix<float64_t>(_X));
     
    bool pass = true; 
    auto y_pred = this->retrieve_labels(features, true, pass);
    features->free_features();
    return y_pred; 
}

VectorXf ML::predict_vector(const MatrixXf& X)
{
    shared_ptr<CLabels> labels = predict(X);
    return labels_to_vector(labels);     
    
}

ArrayXXf ML::predict_proba(const MatrixXf& X)
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

        /* AFAIK, the only reliable way to get the number of classes
         * is to measure the output */
        int n_classes = MLabels->get_multiclass_confidences(0).size();
        MatrixXd confidences(n_classes, 
                             MLabels->get_num_labels()); 
        for (int i =0; i<confidences.cols(); ++i)
        {
            SGVector<double> tmp = \
                           MLabels->get_multiclass_confidences(i);
            confidences.col(i) = Map<ArrayXd>(tmp.data(),tmp.size());
        }

        return confidences.template cast<float>();;
    }
    else
        THROW_INVALID_ARGUMENT("Error: predict_proba not defined for "
                "problem type or ML method");
    return ArrayXXf();
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

float ML::get_bias() const
{   
    // get bias weight. only works with linear machines
    if ( in({L1_LR, LR, LARS, Ridge}, ml_type) )
    {
        // in the multiclass case, return the average bias
        if (this->prob_type == sh::PT_MULTICLASS)
        {
            auto biases = \
            dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(p_est)
            ->get_bias();
            return accumulate(biases.begin(), biases.end(), 0.0)/biases.size();
        }
        else
            return dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_bias();
    }
    else
        return 0;

}

void ML::set_bias(float b)
{
    // set bias weight. only works with linear machines
    if (in({L1_LR, LR, LARS, Ridge}, ml_type) 
        && this->prob_type != sh::PT_MULTICLASS)
    {
        return dynamic_pointer_cast<sh::CLinearMachine>(
                p_est)->set_bias(b);
    }
    else
        THROW_RUNTIME_ERROR("WARNING: Couldn't set bias, "
                "not a binary linear machine");
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
    // map to Eigen vector to check output
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
    FT::Eval::Scorer S(params.scorer_);
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
    return shared_ptr<CLabels>();
}

/// serialization
void to_json(json& j, const ML& ml)
{
    j["ml_type"] =  ml.ml_type;
    j["ml_str"] =  ml.ml_str;
    j["prob_type"] =  ml.prob_type;
    j["N"] =  ml.N;
    j["max_train_time"] =  ml.max_train_time;
    j["normalize"] =  ml.normalize;
    j["C"] =  ml.C;
    // if ml is a linear model, store the weights and bias so it can be reproduced
    // multiclass is handled first. 
    if (in({LARS, Ridge, LR, L1_LR, SVM}, ml.ml_type))
    {
        if (ml.prob_type == PT_MULTICLASS
                && in({LR, L1_LR, SVM}, ml.ml_type) ) 
        {
            vector<SGVector<double>> shogun_weights;
            vector<double> shogun_biases;

            if( in({LR, L1_LR}, ml.ml_type))
            {
                shogun_weights = dynamic_pointer_cast<
                    sh::CMulticlassLogisticRegression>(ml.p_est)->get_w();
                shogun_biases = dynamic_pointer_cast<
                    sh::CMulticlassLogisticRegression>(ml.p_est)->get_bias();
            }
            else //SVM
                shogun_weights = \
                  dynamic_pointer_cast<sh::CMyMulticlassLibLinear>(
                          ml.p_est)->get_w();

            vector<VectorXd> weights;
            for (int i = 0; i < shogun_weights.size(); ++i)
            {
                //TODO: fix this, grab shogun data from underlying array
                weights.push_back(VectorXd());
                weights.at(i) = Map<VectorXd>(shogun_weights.at(i).data(), 
                        shogun_weights.at(i).size());
            } 
            j["w"] = weights;
            j["bias"] = shogun_biases;
        }
        else
        {
            vector<double> weights;
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(
                    ml.p_est)->get_w();
            
            weights.assign(tmp.data(), tmp.data()+tmp.size());          
            j["w"] = weights;
            j["bias"] = ml.get_bias();
        }
        
    }
    else
        WARN("this is not a linear model; at the moment,"
                " it will need to be refit to be used after loading.");
}

void from_json(const json& j, ML& model)
{
    j.at("ml_type").get_to(model.ml_type); 
    j.at("ml_str").get_to(model.ml_str); 
    j.at("prob_type").get_to(model.prob_type); 
    j.at("N").get_to(model.N); 
    j.at("max_train_time").get_to(model.max_train_time); 
    j.at("normalize").get_to(model.normalize); 
    j.at("C").get_to(model.C); 
 
    // initialize the underlying shogun ML model
    model.init(true);
    
    // if model is a linear model, set the weights and bias 
    // so it can be reproduced
    
    if (in({LARS, Ridge, LR, L1_LR, SVM}, model.ml_type))
    {
        if(model.prob_type == PT_MULTICLASS 
                && in({LR, L1_LR, SVM}, model.ml_type) ) 
        {
            // TODO: set biases
            vector<VectorXd> multi_weights = j.at("w");
            if( in({LR, L1_LR}, model.ml_type))
                dynamic_pointer_cast<
                    sh::CMulticlassLogisticRegression>(model.p_est)->set_w(
                            multi_weights);
            else //SVM
                dynamic_pointer_cast<sh::CMyMulticlassLibLinear>(
                          model.p_est)->set_w(multi_weights);
        }
        else
        {
            VectorXd weights; 
            j.at("w").get_to(weights);

            dynamic_pointer_cast<sh::CLinearMachine>(
                    model.p_est)->set_w(sh::SGVector<double>(weights).clone());
            dynamic_pointer_cast<sh::CLinearMachine>(
                    model.p_est)->set_bias(j.at("bias").get<float>());
        }

     }
}

void to_json(json& j, const shared_ptr<ML>& model)
{
    to_json(j, *model);
}

void from_json(const json& j, shared_ptr<ML>& model)
{
    if (model == 0)
        model = shared_ptr<ML>(new ML());
    from_json(j, *model);
}

} // namespace Model
} // namespace FT
