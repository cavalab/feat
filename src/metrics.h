/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef METRICS_H
#define METRIC_H
#include <shogun/labels/Labels.h>

namespace FT{
namespace metrics{

    /* Scoring functions */

    // Squared difference
    VectorXd squared_difference(const VectorXd& y, const VectorXd& yhat)
    {
        return pow(yhat - y, 2);
    }

    // Derivative of squared difference with respec to yhat
    VectorXd d_squared_difference(const VectorXd& y, const VectorXd& yhat) {
        return 2 * (yhat - y);
    }
    
    /// mean squared error
    double mse(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
               const vector<float>& weights=vector<float>() )
    {
        loss = (yhat - y).array().pow(2);
        return loss.mean(); 
    }
    
    double mse_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, 
               const vector<float>& weights=vector<float>() )
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());
        return mse(y,yhat,loss,weights);
    }
    
    /// log loss
    double log_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss,
                      const vector<float>& weights={1.0,1.0})
    {
        double eps = pow(10,-10);

        loss.resize(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) < eps || 1 - yhat(i) < eps)
                // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps))*weights.at(y(i));
            else
                loss(i) = -(y(i)*log(yhat(i)) + (1-y(i))*log(1-yhat(i)))*weights.at(y(i));
        }   
        /* std::cout << "loss: " << loss.transpose() << "\n"; */
        return loss.mean();
    }

    /// log loss
    double log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                      const vector<float>& weights={1.0,1.0})
    {
        dynamic_pointer_cast<sh::CBinaryLabels>(labels)->scores_to_probabilities();
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
        Map<VectorXd> yhat(tmp.data(),tmp.size());
        return log_loss(y,yhat,loss,weights);
    }
 
    /// multinomial log loss
    double multi_log_loss(const VectorXd& y, const ArrayXXd& confidences, VectorXd& loss,
                      const vector<float>& weights=vector<float>())
    {
        // get class labels
        vector<double> uc = unique(y);
        vector<float> w;
        if (weights.empty())
            w = vector<float>(1.0,uc.size());
        else
            w = weights;

        double eps = pow(10,-10);

        loss = VectorXd::Zero(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            for (const auto& c : uc)
            {
                ArrayXd yhat = confidences.col(int(c));
                /* std::cout << "class " << c << "\n"; */

                double yi = y(i) == c ? 1.0 : 0.0 ; 
                /* std::cout << "yi: " << yi << ", yhat(" << i << "): " << yhat(i) ; */  
                if (y(i) == c)
                {
                    if (yhat(i) < eps || 1 - yhat(i) < eps)
                    {
                        // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                        loss(i) += -log(eps);
                        /* std::cout << ", loss(" << i << ") += " << -yi*log(eps); */
                    }
                    else
                    {
                        loss(i) += -log(yhat(i));
                        /* std::cout << ", loss(" << i << ") += " << -yi*log(yhat(i)); */
                    }
                }
                /* std::cout << "\n"; */
            }   
            /* std::cout << "w.at(y(" << i << ")): " << w.at(y(i)) << "\n"; */
            loss(i) = loss(i)*w.at(y(i));
        }
        /* std::cout << "loss: " << loss.transpose() << "\n"; */
        /* std::cout << "mean loss: " << loss.mean() << "\n"; */
        return loss.mean();
    }  

    /// multinomial log loss
    double multi_log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                      const vector<float>& weights=vector<float>())
    {
        /* // get class labels */
        /* vector<double> uc = unique(y); */
        /* vector<float> w; */
        /* if (weights.empty()) */
        /*     w = vector<float>(1.0,uc.size()); */
        /* else */
        /*     w = weights; */

        /* SGVector<double> labs = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)-> */
        /*                                                     get_unique_labels(); */
        /* /1* std::cout << "unique labels: \n"; *1/ */ 
        /* /1* for (int i = 0; i < labs.size(); ++i) std::cout << labs[i] << " " ; std::cout << "\n"; *1/ */

        /* /1* int nclasses = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->get_num_classes(); *1/ */
        /* /1* std::cout << "nclasses: " << nclasses << "\n"; *1/ */

        /* SGVector<double> ypred = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)-> */
        /*                                                     get_labels(); */
        /* Map<ArrayXd> yhat(ypred.data(),y.size()); */
        /* /1* std::cout << "yhat: " << yhat.transpose() << "\n"; *1/ */

        ArrayXXd confidences(y.size(),unique(y).size());
        /* std::cout << "confidences:\n"; */ 
        for (unsigned i =0; i<y.size(); ++i)
        {
            SGVector<double> tmp = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                                                            get_multiclass_confidences(int(i));
            confidences.row(i) = Map<ArrayXd>(tmp.data(),tmp.size());
            /* std::cout << confidences.row(i) << "\n"; */
        }
        /* for (auto c : uc) */
        /*     std::cout << "class " << int(c) << ": " << confidences.col(int(c)).transpose() << "\n"; */
        /* std::cout << "in log loss\n"; */

        return multi_log_loss(y, confidences, loss, weights); 

    }

    /// 1 - balanced accuracy 
    double bal_zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
               const vector<float>& weights=vector<float>() )
    {
        vector<double> uc = unique(y);
        vector<int> c;
        for (const auto& i : uc)
            c.push_back(int(i));
         
        // sensitivity (TP) and specificity (TN)
        vector<double> TP(c.size(),0.0), TN(c.size(), 0.0), P(c.size(),0.0), N(c.size(),0.0);
        ArrayXd class_accuracies(c.size());
       
        // get class counts
        
        for (unsigned i=0; i< c.size(); ++i)
        {
            P.at(i) = (y.array().cast<int>() == c[i]).count();  // total positives for this class
            N.at(i) = (y.array().cast<int>() != c[i]).count();  // total negatives for this class
        }
        

        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) == y(i))                    // true positive
                ++TP.at(y(i) == -1 ? 0 : y(i));     // if-then ? accounts for -1 class encoding

            for (unsigned j = 0; j < c.size(); ++j)
                if ( y(i) !=c.at(j) && yhat(i) != c.at(j) )    // true negative
                    ++TN.at(j);    
            
        }

        // class-wise accuracy = 1/2 ( true positive rate + true negative rate)
        for (unsigned i=0; i< c.size(); ++i){
            class_accuracies(i) = (TP[i]/P[i] + TN[i]/N[i])/2; 
            //std::cout << "TP(" << i << "): " << TP[i] << ", P[" << i << "]: " << P[i] << "\n";
            //std::cout << "TN(" << i << "): " << TN[i] << ", N[" << i << "]: " << N[i] << "\n";
            //std::cout << "class accuracy(" << i << "): " << class_accuracies(i) << "\n";
        }
       
        // set loss vectors if third argument supplied
        loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();

        return 1.0 - class_accuracies.mean();
    }

    double bal_zero_one_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, 
                                   VectorXd& loss, const vector<float>& weights=vector<float>() )
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return bal_zero_one_loss(y, yhat, loss, weights);
    }

    double zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
               const vector<float>& weights=vector<float>() )
    {
        loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
        //TODO: weight loss by sample weights
        return loss.mean();
    }
    
    /// 1 - accuracy 
    double zero_one_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, 
               const vector<float>& weights=vector<float>() )
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return zero_one_loss(y,yhat,loss,weights);
    }
   

    /* double bal_log_loss(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, */ 
    /*            const vector<float>& weights=vector<float>() ) */
    /* { */
      
    /*     loss = log_loss(y,yhat); */

    /*     vector<double> uc = unique(y); */
    /*     vector<int> c; */ 
    /*     for (const auto& i : uc) */
    /*         c.push_back(int(i)); */
        
    /*     vector<double> class_loss(c.size(),0); */

    /*     for (unsigned i = 0; i < c.size(); ++i) */
    /*     { */
    /*         int n = (y.cast<int>().array() == c[i]).count(); */
    /*         class_loss[i] = (y.cast<int>().array() == c[i]).select(loss.array(),0).sum()/n; */
        
    /*     } */
    /*     // return balanced class losses */ 
    /*     Map<ArrayXd> cl(class_loss.data(),class_loss.size()); */        
    /*     return cl.mean(); */
    /* } */
} // metrics
} // FT


#endif

