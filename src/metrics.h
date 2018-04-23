/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef METRICS_H
#define METRIC_H

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
    
    /// log loss
    VectorXd log_loss(const VectorXd& y, const VectorXd& yhat)
    {
        /* std::cout << "in log loss\n"; */
        double eps = pow(10,-10);

        VectorXd loss(y.rows());  
        for (unsigned i = 0; i < y.rows(); ++i)
        {
            if (yhat(i) < eps || 1 - yhat(i) < eps)
                // clip probabilities since log loss is undefined for yhat=0 or yhat=1
                loss(i) = -(y(i)*log(eps) + (1-y(i))*log(1-eps));
            else
                loss(i) = -(y(i)*log(yhat(i)) + (1-y(i))*log(1-yhat(i)));
        }   
        return loss;
    }

    
    /// mean squared error
    double mse(const VectorXd& y, const VectorXd& yhat, VectorXd& loss )
    {
        loss = (yhat - y).array().pow(2);
        return loss.mean(); 
    };


    double bal_zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss )
    {
        /* std::cout << "in bal_zero_one_loss\n"; */
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

    double bal_log_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss )
    {
      
        loss = log_loss(y,yhat);

        vector<double> uc = unique(y);
        vector<int> c; 
        for (const auto& i : uc)
            c.push_back(int(i));
        
        vector<double> class_loss(c.size(),0);

        for (unsigned i = 0; i < c.size(); ++i)
        {
            int n = (y.cast<int>().array() == c[i]).count();
            class_loss[i] = (y.cast<int>().array() == c[i]).select(loss.array(),0).sum()/n;
        
        }
        // return balanced class losses 
        Map<ArrayXd> cl(class_loss.data(),class_loss.size());        
        return cl.mean();
    }
    
    /// 1 - accuracy 
    double zero_one_loss(const VectorXd& y, const VectorXd& yhat, VectorXd& loss )
    {
        loss = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
        return loss.mean();
    }
   
    

} // metrics
} // FT


#endif

