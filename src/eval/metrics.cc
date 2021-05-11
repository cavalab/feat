/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "metrics.h"

namespace FT
{
    
    using namespace Util;
        
    namespace Eval
    {
        /* Scoring functions */

        // Squared difference
        VectorXf squared_difference(const VectorXf& y, const VectorXf& yhat)
        {
            return (yhat - y).array().pow(2);
        }

        VectorXf squared_difference(const VectorXf& y, 
                shared_ptr<CLabels>& labels,  
                const vector<float>& weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            
            Map<VectorXf> yhat(tmp.data(),tmp.size());
            return squared_difference(y,yhat);
            /* if (weights.empty()) */
            /*     return (yhat - y).array().pow(2); */
            /* else */
            /* { */
            /*     assert(weights.size() == yhat.size()); */
            /*     ArrayXf w = ArrayXf::Map(weights.data(),weights.size()); */
            /*     return (yhat - y).array().pow(2) * w.cast<float>() ; */
            /* } */
        }

        // Derivative of squared difference with respec to yhat
        VectorXf d_squared_difference(const VectorXf& y, const VectorXf& yhat)
        {
            return 2 * (yhat - y);
        }
       
        // Derivative of squared difference with respec to yhat
        VectorXf d_squared_difference(const VectorXf& y, 
                shared_ptr<CLabels>& labels, const vector<float>& weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());

            return d_squared_difference(y,yhat);
            /* if (weights.empty()) */
            /*     return 2 * (yhat - y); */
            /* else */
            /* { */
            /*     assert(weights.size() == yhat.size()); */
            /*     ArrayXf w = ArrayXf::Map(weights.data(),weights.size()); */
            /*     return 2 * (yhat - y).array() * w.cast<float>() ; */
            /* } */
        }

        /// mean squared error
        float mse(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& weights)
        {
            loss = (yhat - y).array().pow(2);
            return loss.mean(); 
        }
        
        float mse_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss, 
                const vector<float>& weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());
            return mse(y,yhat,loss,weights);
        }
     
        VectorXf log_loss(const VectorXf& y, const VectorXf& yhat, 
                          const vector<float>& class_weights)
        {
            float eps = pow(10,-10);
            
            VectorXf loss;
            
            float sum_weights = 0; 
            loss.resize(y.rows());  
            for (unsigned i = 0; i < y.rows(); ++i)
            {
                if (yhat(i) < eps || 1 - yhat(i) < eps)
                    // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                    loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
                else
                    loss(i) = -(y(i)*log(yhat(i)) + (1-y(i))*log(1-yhat(i)));
                if (loss(i)<0)
                    THROW_RUNTIME_ERROR("loss(i)= " + to_string(loss(i)) 
                            + ". y = " + to_string(y(i)) + ", yhat(i) = " 
                            + to_string(yhat(i)));

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
       
        VectorXf log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            
            Map<VectorXf> yhat(tmp.data(),tmp.size());
           
            VectorXf loss = log_loss(y,yhat,class_weights);
            return loss; 
        }
        
        /// log loss
        float mean_log_loss(const VectorXf& y, 
                const VectorXf& yhat, VectorXf& loss,
                const vector<float>& class_weights)
        {
               
            /* std::cout << "loss: " << loss.transpose() << "\n"; */
            loss = log_loss(y,yhat,class_weights);
            return loss.mean();
        }

        /// log loss
        float log_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss,
                const vector<float>& class_weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());
            /* cout << "log loss yhat: " << yhat.transpose() << "\n"; */
            return mean_log_loss(y,yhat,loss,class_weights);
        }

        VectorXf d_log_loss(const VectorXf& y, const VectorXf& yhat, 
                            const vector<float>& class_weights)
        {
            VectorXf dll(y.size()); 
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

        VectorXf d_log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                            const vector<float>& class_weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());

            return d_log_loss(y,yhat,class_weights);
        }

        /* float weighted_average(const VectorXf& y, const VectorXf& w) */
        /* { */
        /*     w = w.array() / w.sum(); */
        /*     return (y.array() * w.array()).mean(); */
        /* } */
        /// multinomial log loss
        VectorXf multi_log_loss(const VectorXf& y, const ArrayXXf& confidences, 
                const vector<float>& class_weights)
        {
            VectorXf loss = VectorXf::Zero(y.rows());  
            
            // get class labels
            vector<float> uc = unique(y);

            float eps = pow(10,-10);
            float sum_weights = 0; 
            for (unsigned i = 0; i < y.rows(); ++i)
            {
                for (const auto& c : uc)
                {
                    ArrayXf yhat = confidences.col(int(c));
                    /* std::cout << "class " << c << "\n"; */

                    /* float yi = y(i) == c ? 1.0 : 0.0 ; */ 
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

        float mean_multi_log_loss(const VectorXf& y, 
                const ArrayXXf& confidences, VectorXf& loss,
                const vector<float>& class_weights)
        {
            loss = multi_log_loss(y, confidences, class_weights);
     
            /* std::cout << "loss: " << loss.transpose() << "\n"; */
            /* std::cout << "mean loss: " << loss.mean() << "\n"; */
            return loss.mean();
        }  

        /// multinomial log loss
        float multi_log_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss,
                const vector<float>& class_weights)
        {
            ArrayXXf confidences(y.size(),unique(y).size());
            /* std::cout << "confidences:\n"; */ 
            for (unsigned i =0; i<y.size(); ++i)
            {
                SGVector<double> _tmp = 
                    dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                                            get_multiclass_confidences(int(i));
                SGVector<float> tmp(_tmp.begin(), _tmp.end());
                if (confidences.cols() != tmp.size())
                {
                    /* WARN("mismatch in confidences: expected " */
                    /*      + to_string(confidences.cols()) */ 
                    /*      + " values, got " */
                    /*      + to_string(tmp.size()) */ 
                    /*      + " from labels"); */
                    confidences.row(i) = 0;
                }
                else
                    confidences.row(i) = Map<ArrayXf>(tmp.data(),tmp.size());
                /* std::cout << confidences.row(i) << "\n"; */
            }
            /* for (auto c : uc) */
            /*     std::cout << "class " << int(c) << ": " << confidences.col(int(c)).transpose() << "\n"; */
            /* std::cout << "in log loss\n"; */

            return mean_multi_log_loss(y, confidences, loss, class_weights); 

        }

        VectorXf multi_log_loss(const VectorXf& y, shared_ptr<CLabels>& labels, 
                          const vector<float>& class_weights)
        {
            ArrayXXf confidences(y.size(),unique(y).size());
            /* std::cout << "confidences:\n"; */ 
            for (unsigned i =0; i<y.size(); ++i)
            {
                SGVector<double> _tmp = 
                    dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                    get_multiclass_confidences(int(i));

                SGVector<float> tmp(_tmp.begin(), _tmp.end());
                confidences.row(i) = Map<ArrayXf>(tmp.data(),tmp.size());
                /* std::cout << confidences.row(i) << "\n"; */
            }
            
            VectorXf loss = multi_log_loss(y, confidences, class_weights);
            return loss;        
        }
        /// derivative of multinomial log loss
        VectorXf d_multi_log_loss(const VectorXf& y, 
                shared_ptr<CLabels>& labels, 
                const vector<float>& class_weights)
        {
            ArrayXXf confidences(y.size(),unique(y).size());
            /* std::cout << "confidences:\n"; */ 
            for (unsigned i =0; i<y.size(); ++i)
            {
                SGVector<double> _tmp = 
                    dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->
                    get_multiclass_confidences(int(i));
                SGVector<float> tmp(_tmp.begin(), _tmp.end());
                confidences.row(i) = Map<ArrayXf>(tmp.data(),tmp.size());
                /* std::cout << confidences.row(i) << "\n"; */
            }
           
            VectorXf loss = VectorXf::Zero(y.rows());  
            
            // get class labels
            vector<float> uc = unique(y);

            float eps = pow(10,-10);
            float sum_weights = 0; 
            for (unsigned i = 0; i < y.rows(); ++i)
            {
                for (const auto& c : uc)
                {
                    ArrayXf yhat = confidences.col(int(c));
                    /* std::cout << "class " << c << "\n"; */

                    /* float yi = y(i) == c ? 1.0 : 0.0 ; */ 
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
        float bal_zero_one_loss(const VectorXf& y, const VectorXf& yhat, 
                VectorXf& loss, const vector<float>& class_weights)
        {
            vector<float> uc = unique(y);
            vector<int> c;
            for (const auto& i : uc)
                c.push_back(int(i));
             
            // sensitivity (TP) and specificity (TN)
            vector<float> TP(c.size(),0.0), TN(c.size(), 0.0), P(c.size(),0.0), N(c.size(),0.0);
            ArrayXf class_accuracies(c.size());
           
            // get class counts
            
            for (unsigned i=0; i< c.size(); ++i)
            {
                P.at(i) = (y.array().cast<int>() == c.at(i)).count();  // total positives for this class
                N.at(i) = (y.array().cast<int>() != c.at(i)).count();  // total negatives for this class
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
                class_accuracies(i) = (TP.at(i)/P.at(i) + TN.at(i)/N.at(i))/2; 
                //std::cout << "TP(" << i << "): " << TP.at(i) << ", P[" << i << "]: " << P.at(i) << "\n";
                //std::cout << "TN(" << i << "): " << TN.at(i) << ", N[" << i << "]: " << N.at(i) << "\n";
                //std::cout << "class accuracy(" << i << "): " << class_accuracies(i) << "\n";
            }
           
            // set loss vectors if third argument supplied
            loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<float>();

            return 1.0 - class_accuracies.mean();
        }

        float bal_zero_one_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, 
                VectorXf& loss, const vector<float>& class_weights)
        {
        
            SGVector<double> _tmp;
            
            auto ptrmulticast = dynamic_pointer_cast<sh::CMulticlassLabels>(labels);
            
            if(ptrmulticast == NULL)
            {
                auto ptrbinary = dynamic_pointer_cast<sh::CBinaryLabels>(labels);
                _tmp = ptrbinary->get_labels();
            }
            else
                _tmp = ptrmulticast->get_labels();
                
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());

            return bal_zero_one_loss(y, yhat, loss, class_weights);
        }

        float zero_one_loss(const VectorXf& y, const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& class_weights)
        {
            loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<float>();
            //TODO: weight loss by sample weights
            return loss.mean();
        }
        
        /// 1 - accuracy 
        float zero_one_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss, 
                const vector<float>& class_weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());

            return zero_one_loss(y,yhat,loss,class_weights);
        }
        // make a false positive rate loss
        //
        float false_positive_loss(const VectorXf& y, 
                const VectorXf& yhat, VectorXf& loss, 
                   const vector<float>& class_weights)
        {
            ArrayXb ybool = y.cast<bool>();
            /* ArrayXb noty = !ybool; */ 
            loss = (yhat.cast<bool>().select(
                        !ybool, false)).cast<float>();
            return loss.sum()/float((y.size() - ybool.count()));
        }
        /// false positive rate
        float false_positive_loss_label(const VectorXf& y, 
                const shared_ptr<CLabels>& labels, VectorXf& loss, 
                const vector<float>& class_weights)
        {
            SGVector<double> _tmp = 
                dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
            SGVector<float> tmp(_tmp.begin(), _tmp.end());
            Map<VectorXf> yhat(tmp.data(),tmp.size());

            return false_positive_loss(y,yhat,loss,class_weights);
        }
    } // metrics
} // FT


