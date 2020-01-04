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
    namespace Eval
    {

        /* Scoring functions */

        // Squared difference
        VectorXf squared_difference(const VectorXf& y, const VectorXf& yhat);

        VectorXf squared_difference(const VectorXf& y, shared_ptr<CLabels>& labels,  
                   const vector<float>& weights=vector<float>());

        // Derivative of squared difference with respec to yhat
        VectorXf d_squared_difference(const VectorXf& y, const VectorXf& yhat);
       
        // Derivative of squared difference with respec to yhat
        VectorXf d_squared_difference(const VectorXf& y, shared_ptr<CLabels>& labels,
                               const vector<float>& weights=vector<float>());

        /// mean squared error
        float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& weights=vector<float>() );
        
        float mse_label(const VectorXf& y, const shared_ptr<CLabels>& labels, VectorXf& loss, 
                   const vector<float>& weights=vector<float>() );
     
        VectorXf log_loss(const VectorXf& y, const VectorXf& yhat, 
                          const vector<float>& class_weights=vector<float>());
       
        VectorXf log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
        
        /// log loss
        float mean_log_loss(const VectorXf& y, const VectorXf& yhat, VectorXf& loss,
                          const vector<float>& class_weights = vector<float>());
        /// log loss
        float log_loss_label(const VectorXf& y, const shared_ptr<CLabels>& labels, VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

        VectorXf d_log_loss(const VectorXf& y, const VectorXf& yhat, 
                            const vector<float>& class_weights=vector<float>());

        VectorXf d_log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                            const vector<float>& class_weights=vector<float>());

        /* float weighted_average(const VectorXf& y, const VectorXf& w) */
        /* { */
        /*     w = w.array() / w.sum(); */
        /*     return (y.array() * w.array()).mean(); */
        /* } */
        /// multinomial log loss
        VectorXf multi_log_loss(const VectorXf& y, const ArrayXXf& confidences, 
                const vector<float>& class_weights=vector<float>());

        float mean_multi_log_loss(const VectorXf& y, const ArrayXXf& confidences, VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

        /// multinomial log loss
        float multi_log_loss_label(const VectorXf& y, const shared_ptr<CLabels>& labels, VectorXf& loss,
                          const vector<float>& class_weights=vector<float>());

        VectorXf multi_log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
                          
        /// derivative of multinomial log loss
        VectorXf d_multi_log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights=vector<float>());
                          
        /// 1 - balanced accuracy 
        float bal_zero_one_loss(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& class_weights=vector<float>() );

        float bal_zero_one_loss_label(const VectorXf& y, const shared_ptr<CLabels>& labels, 
                                       VectorXf& loss, const vector<float>& class_weights=vector<float>() );

        float zero_one_loss(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& class_weights=vector<float>() );
        
        /// 1 - accuracy 
        float zero_one_loss_label(const VectorXf& y, const shared_ptr<CLabels>& labels, VectorXf& loss, 
                   const vector<float>& class_weights=vector<float>() );

        /* float bal_log_loss(const VectorXf& y, const shared_ptr<CLabels>& labels, VectorXf& loss, */ 
        /*            const vector<float>& weights=vector<float>() ) */
        /* { */
          
        /*     loss = log_loss(y,yhat); */

        /*     vector<float> uc = unique(y); */
        /*     vector<int> c; */ 
        /*     for (const auto& i : uc) */
        /*         c.push_back(int(i)); */
            
        /*     vector<float> class_loss(c.size(),0); */

        /*     for (unsigned i = 0; i < c.size(); ++i) */
        /*     { */
        /*         int n = (y.cast<int>().array() == c[i]).count(); */
        /*         class_loss[i] = (y.cast<int>().array() == c[i]).select(loss.array(),0).sum()/n; */
            
        /*     } */
        /*     // return balanced class losses */ 
        /*     Map<ArrayXf> cl(class_loss.data(),class_loss.size()); */        
        /*     return cl.mean(); */
        /* } */
        float false_positive_loss(const VectorXf& y, 
                const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& class_weights);
        float false_positive_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss, 
                const vector<float>& class_weights);
    } // metrics
} // FT

#endif

