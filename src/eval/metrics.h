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
#include "../util/utils.h"
namespace sh = shogun;
using sh::CLabels;
using sh::SGVector; 
using Eigen::Map;
using Eigen::ArrayXf;

namespace FT
{
    namespace metrics
    {

        /* Scoring functions */

        // Squared difference
        VectorXd squared_difference(const VectorXd& y, const VectorXd& yhat);

        VectorXd squared_difference(const VectorXd& y, shared_ptr<CLabels>& labels,  
                   const vector<float>& weights=vector<float>());

        // Derivative of squared difference with respec to yhat
        VectorXd d_squared_difference(const VectorXd& y, const VectorXd& yhat);
       
        // Derivative of squared difference with respec to yhat
        VectorXd d_squared_difference(const VectorXd& y, shared_ptr<CLabels>& labels,
                               const vector<float>& weights=vector<float>());

        /// mean squared error
        double mse(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
                   const vector<float>& weights=vector<float>() );
        
        double mse_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, 
                   const vector<float>& weights=vector<float>() );
     
        VectorXd log_loss(const VectorXd& y, const VectorXd& yhat, 
                          const vector<float>& class_weights=vector<float>());
       
        VectorXd log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
        
        /// log loss
        double mean_log_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss,
                          const vector<float>& class_weights = vector<float>());
        /// log loss
        double log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                          const vector<float>& class_weights=vector<float>());

        VectorXd d_log_loss(const VectorXd& y, const VectorXd& yhat, 
                            const vector<float>& class_weights=vector<float>());

        VectorXd d_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                            const vector<float>& class_weights=vector<float>());

        /* double weighted_average(const VectorXd& y, const VectorXd& w) */
        /* { */
        /*     w = w.array() / w.sum(); */
        /*     return (y.array() * w.array()).mean(); */
        /* } */
        /// multinomial log loss
        VectorXd multi_log_loss(const VectorXd& y, const ArrayXXd& confidences, 
                const vector<float>& class_weights=vector<float>());

        double mean_multi_log_loss(const VectorXd& y, const ArrayXXd& confidences, VectorXd& loss,
                          const vector<float>& class_weights=vector<float>());

        /// multinomial log loss
        double multi_log_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss,
                          const vector<float>& class_weights=vector<float>());

        VectorXd multi_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
                          
        /// derivative of multinomial log loss
        VectorXd d_multi_log_loss(const VectorXd& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
                          
        /// 1 - balanced accuracy 
        double bal_zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
                   const vector<float>& class_weights=vector<float>() );

        double bal_zero_one_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, 
                                       VectorXd& loss, const vector<float>& class_weights=vector<float>() );

        double zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss, 
                   const vector<float>& class_weights=vector<float>() );
        
        /// 1 - accuracy 
        double zero_one_loss_label(const VectorXd& y, const shared_ptr<CLabels>& labels, VectorXd& loss, 
                   const vector<float>& class_weights=vector<float>() );

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

