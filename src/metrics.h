/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef METRICS_H
#define METRICS_H
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>                                                         
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include "utils.h"
namespace sh = shogun;
using sh::CLabels;
using sh::SGVector; 
using Eigen::Map;
using Eigen::ArrayXf;

namespace FT{
namespace metrics{

    /* Scoring functions */

    // Squared difference
    VectorXd squared_difference(const VectorXd& y, const VectorXd& yhat)
    {
        return (yhat - y).array().pow(2);
    }

    VectorXd squared_difference(const VectorXd& y, shared_ptr<CLabels>& labels,  
               const vector<float>& weights=vector<float>())
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());
        return squared_difference(y,yhat);
        /* if (weights.empty()) */
        /*     return (yhat - y).array().pow(2); */
        /* else */
        /* { */
        /*     assert(weights.size() == yhat.size()); */
        /*     ArrayXf w = ArrayXf::Map(weights.data(),weights.size()); */
        /*     return (yhat - y).array().pow(2) * w.cast<double>() ; */
        /* } */
    }

    // Derivative of squared difference with respec to yhat
    VectorXd d_squared_difference(const VectorXd& y, const VectorXd& yhat)
    {
        return 2 * (yhat - y);
    }
   
    // Derivative of squared difference with respec to yhat
    VectorXd d_squared_difference(const VectorXd& y, shared_ptr<CLabels>& labels,
                           const vector<float>& weights=vector<float>())
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return d_squared_difference(y,yhat);
        /* if (weights.empty()) */
        /*     return 2 * (yhat - y); */
        /* else */
        /* { */
        /*     assert(weights.size() == yhat.size()); */
        /*     ArrayXf w = ArrayXf::Map(weights.data(),weights.size()); */
        /*     return 2 * (yhat - y).array() * w.cast<double>() ; */
        /* } */
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
 
    VectorXd log_loss(const VectorXd& y, const VectorXd& yhat, 
                      const vector<float>& class_weights=vector<float>())
    {
        double eps = pow(10,-10);
        
        VectorXd loss;
        
        double sum_weights = 0; 
        loss.resize(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) < eps || 1 - yhat(i) < eps)
                // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
            else
                loss(i) = -(y(i)*log(yhat(i)) + (1-y(i))*log(1-yhat(i)));
            if (loss(i)<0)
                HANDLE_ERROR_THROW("loss(i)= " + to_string(loss(i)) + ". y = " + to_string(y(i)) + ", yhat(i) = " + to_string(yhat(i)));

            if (!class_weights.empty())
            {
                loss(i) = loss(i) * class_weights.at(y(i));
                sum_weights += class_weights.at(y(i));
            }
        }
        
        if (sum_weights > 0)
            loss = loss.array() / sum_weights * y.size(); // normalize weight contributions
        
        return loss;
    }   
   
    VectorXd log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                      const vector<float>& class_weights=vector<float>())
    {
        /* dynamic_pointer_cast<sh::CBinaryLabels>(labels)->scores_to_probabilities(); */
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
        Map<VectorXd> yhat(tmp.data(),tmp.size());
       
        VectorXd loss = log_loss(y,yhat,class_weights);
        return loss; 
        /* if (weights.empty()) */
        /*     return loss; */
        /* else */
        /* { */
        /*     ArrayXb w(y.size()); */
        /*     for (int i = 0; i < class_weights.size(); ++i) */
        /*     { */
        /*         w = (y.cast<int>() == i).select(class_weights[i], w); */
        /*     } */
        /*     cout << "w: " << w.transpose(); */ 
        /*     return loss.array() * w; */
        /* } */
    }
    
    /// log loss
    double mean_log_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss,
                      const vector<float>& class_weights = vector<float>())
    {
           
        /* std::cout << "loss: " << loss.transpose() << "\n"; */
        loss = log_loss(y,yhat,class_weights);
        if (loss.mean() < 0)
        {
            cout << "LOG LOSS MEAN < 0 !!!!!!\n";
            cout << "loss: " << loss.transpose() << "\n";
            cout << "y: " << y.transpose() << "\n";
            cout << "yhat: " << yhat.transpose() << "\n";

        }
        /* if (!sample_weights.empty()) */
        /* { */
        /*     ArrayXf sw = ArrayXf::Map(sample_weights.data(), sample_weights.size()); */
        /*     if (sw.size() != loss.size() ){ */
        /*         std::cerr << "Error: sample_weights size(" << sw.size() << ") different than" */
        /*                   << " loss size(" << loss.size() << ")\n"; */
        /*         std::cout << "y size: " << y.size() <<"\n"; */
        /*         std::cout << "yhat size: "<< yhat.size() << "\n"; */
        /*         exit(1); */
        /*     } */
        /*     loss = loss.array() * sw.cast<double>() ; */ 
        /*     if ((sw < 0).any()) */
        /*     { */
        /*         cout << "NEGATIVE SAMPLE WEIGHTS\n"; */
        /*         cout << "sample_weights: " << sw.transpose() << "\n"; */
        /*     } */
        /* } */
        
        /* if (loss.mean() < 0) */
        /* { */
        /*     cout << "LOG LOSS MEAN < 0 !!!!!!\n"; */
        /*     cout << "loss: " << loss.transpose() << "\n"; */
        /*     cout << "y: " << y.transpose() << "\n"; */
        /*     cout << "yhat: " << yhat.transpose() << "\n"; */

        /* } */
        return loss.mean();
    }

    /// log loss
    double log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                      const vector<float>& class_weights=vector<float>())
    {
        /* dynamic_pointer_cast<sh::CBinaryLabels>(labels)->scores_to_probabilities(); */
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
        Map<VectorXd> yhat(tmp.data(),tmp.size());
        /* cout << "log loss yhat: " << yhat.transpose() << "\n"; */
        return mean_log_loss(y,yhat,loss,class_weights);
    }

    VectorXd d_log_loss(const VectorXd& y, const VectorXd& yhat, 
                        const vector<float>& class_weights=vector<float>())
    {
        VectorXd dll(y.size()); 
        for (int i = 0; i < y.size(); ++i)
        {
            /* dll(i) = (-(yhat(i) - y(i)) / ( (yhat(i)-1)*yhat(i) ) ); */
            // note this derivative assumes a logistic form for yhat, i.e. yhat = 1/(1+exp(-o))
            dll(i) = (yhat(i) - y(i));
            if (!class_weights.empty())
                dll(i) = dll(i) * class_weights.at(y(i));
        }
        return dll;
    }

    VectorXd d_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                        const vector<float>& class_weights=vector<float>())
    {
        /* dynamic_pointer_cast<sh::CBinaryLabels>(labels)->scores_to_probabilities(); */
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return d_log_loss(y,yhat,class_weights);
    }

    /* double weighted_average(const VectorXd& y, const VectorXd& w) */
    /* { */
    /*     w = w.array() / w.sum(); */
    /*     return (y.array() * w.array()).mean(); */
    /* } */
    /// multinomial log loss
    VectorXd multi_log_loss(const VectorXd& y, const ArrayXXd& confidences, 
            const vector<float>& class_weights=vector<float>())
    {
        VectorXd loss = VectorXd::Zero(y.rows());  
        
        // get class labels
        vector<double> uc = unique(y);

        double eps = pow(10,-10);
        double sum_weights = 0; 
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            for (const auto& c : uc)
            {
                ArrayXd yhat = confidences.col(int(c));
                /* std::cout << "class " << c << "\n"; */

                /* double yi = y(i) == c ? 1.0 : 0.0 ; */ 
                /* std::cout << "yi: " << yi << ", yhat(" << i << "): " << yhat(i) ; */  
                if (y(i) == c)
                {
                    if (yhat(i) < eps || 1 - yhat(i) < eps)
                    {
                        // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                        loss(i) += -log(eps);
                    }
                    else
                    {
                        loss(i) += -log(yhat(i));
                    }
                    /* std::cout << ", loss(" << i << ") = " << loss(i); */
                }
                /* std::cout << "\n"; */
             }
            if (!class_weights.empty()){
                /* std::cout << "weights.at(y(" << i << ")): " << class_weights.at(y(i)) << "\n"; */
                loss(i) = loss(i)*class_weights.at(y(i));
                sum_weights += class_weights.at(y(i));
            }
        }
        if (sum_weights > 0)
            loss = loss.array() / sum_weights * y.size(); 
        /* cout << "loss.mean(): " << loss.mean() << "\n"; */
        /* cout << "loss.sum(): " << loss.sum() << "\n"; */
        return loss;
    }

    double mean_multi_log_loss(const VectorXd& y, const ArrayXXd& confidences, VectorXd& loss,
                      const vector<float>& class_weights=vector<float>())
    {
        loss = multi_log_loss(y, confidences, class_weights);
 
        /* std::cout << "loss: " << loss.transpose() << "\n"; */
        /* std::cout << "mean loss: " << loss.mean() << "\n"; */
        return loss.mean();
    }  

    /// multinomial log loss
    double multi_log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                      const vector<float>& class_weights=vector<float>())
    {
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

        return mean_multi_log_loss(y, confidences, loss, class_weights); 

    }

    VectorXd multi_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                      const vector<float>& class_weights=vector<float>())
    {
        ArrayXXd confidences(y.size(),unique(y).size());
        /* std::cout << "confidences:\n"; */ 
        for (unsigned i =0; i<y.size(); ++i)
        {
            SGVector<double> tmp = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                                                            get_multiclass_confidences(int(i));
            confidences.row(i) = Map<ArrayXd>(tmp.data(),tmp.size());
            /* std::cout << confidences.row(i) << "\n"; */
        }
        
        VectorXd loss = multi_log_loss(y, confidences, class_weights);
        return loss;        
    }
    /// derivative of multinomial log loss
    VectorXd d_multi_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                      const vector<float>& class_weights=vector<float>())
    {
        ArrayXXd confidences(y.size(),unique(y).size());
        /* std::cout << "confidences:\n"; */ 
        for (unsigned i =0; i<y.size(); ++i)
        {
            SGVector<double> tmp = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                                                            get_multiclass_confidences(int(i));
            confidences.row(i) = Map<ArrayXd>(tmp.data(),tmp.size());
            /* std::cout << confidences.row(i) << "\n"; */
        }
       
        VectorXd loss = VectorXd::Zero(y.rows());  
        
        // get class labels
        vector<double> uc = unique(y);

        double eps = pow(10,-10);
        double sum_weights = 0; 
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            for (const auto& c : uc)
            {
                ArrayXd yhat = confidences.col(int(c));
                /* std::cout << "class " << c << "\n"; */

                /* double yi = y(i) == c ? 1.0 : 0.0 ; */ 
                /* std::cout << "yi: " << yi << ", yhat(" << i << "): " << yhat(i) ; */  
                if (y(i) == c)
                {
                    if (yhat(i) < eps) 
                    {
                        // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                        loss(i) += -1/eps;
                        /* std::cout << ", loss(" << i << ") += " << -yi*log(eps); */
                    }
                    else
                    {
                        loss(i) += -1/yhat(i);
                        /* std::cout << ", loss(" << i << ") += " << -yi*log(yhat(i)); */
                    }
                }
                /* std::cout << "\n"; */
            }   
            if (!class_weights.empty())
            {
            /* std::cout << "w.at(y(" << i << ")): " << w.at(y(i)) << "\n"; */
                loss(i) = loss(i)*class_weights.at(y(i));
                sum_weights += class_weights.at(y(i));
           /* std::cout << "w.at(y(" << i << ")): " << w.at(y(i)) << "\n"; */
            /* loss(i) = loss(i)*w.at(i); */
            }
        }
        if (sum_weights > 0)
            loss = loss.array() / sum_weights * y.size(); // normalize weight contributions

        return loss;
    }
    /// 1 - balanced accuracy 
    double bal_zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
               const vector<float>& class_weights=vector<float>() )
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
                                   VectorXd& loss, const vector<float>& class_weights=vector<float>() )
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return bal_zero_one_loss(y, yhat, loss, class_weights);
    }

    double zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
               const vector<float>& class_weights=vector<float>() )
    {
        loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
        //TODO: weight loss by sample weights
        return loss.mean();
    }
    
    /// 1 - accuracy 
    double zero_one_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, 
               const vector<float>& class_weights=vector<float>() )
    {
        SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
        Map<VectorXd> yhat(tmp.data(),tmp.size());

        return zero_one_loss(y,yhat,loss,class_weights);
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

