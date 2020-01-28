/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "utils.h"
#include "rnd.h"
#include <unordered_set>

namespace FT{

    namespace Util{

        string PBSTR = "====================";
        int PBWIDTH = 20;
     
        /// limits node output to be between MIN_FLT and MAX_FLT
        void clean(ArrayXf& x)
        {
            x = (x < MIN_FLT).select(MIN_FLT,x);
            x = (isinf(x)).select(MAX_FLT,x);
            x = (isnan(x)).select(0,x);
        };  

        std::string ltrim(std::string str, const std::string& chars)
        {
            str.erase(0, str.find_first_not_of(chars));
            return str;
        }
         
        std::string rtrim(std::string str, const std::string& chars)
        {
            str.erase(str.find_last_not_of(chars) + 1);
            return str;
        }
         
        std::string trim(std::string str, const std::string& chars)
        {
            return ltrim(rtrim(str, chars), chars);
        }

        /// determines data types of columns of matrix X.
        vector<char> find_dtypes(MatrixXf &X)
        {
	        vector<char> dtypes;
	        
	        // get feature types (binary or continuous/categorical)
            int i, j;
            bool isBinary;
            bool isCategorical;
            std::map<float, bool> uniqueMap;
            for(i = 0; i < X.rows(); i++)
            {
                isBinary = true;
                isCategorical = true;
                uniqueMap.clear();
                
                for(j = 0; j < X.cols(); j++)
                {
                    if(X(i, j) != 0 && X(i, j) != 1)
                        isBinary = false;
                    if(X(i,j) != floor(X(i, j)) && X(i,j) != ceil(X(i,j)))
                        isCategorical = false;
                    else
                        uniqueMap[X(i, j)] = true;
                }
            
                if(isBinary)
                    dtypes.push_back('b');
                else
                {
                    if(isCategorical && uniqueMap.size() < 10)
                        dtypes.push_back('c');    
                    else
                        dtypes.push_back('f');
                }
            }
            /* cout << "dtypes: " ; */ 
            /* for (const auto& dt : dtypes) */
            /*     cout << dt << ", "; */
            /* cout << "\n"; */
            return dtypes;

	    }
        
        /// calculate median
        float median(const ArrayXf& v) 
        {
            // instantiate a vector
            vector<float> x(v.size());
            x.assign(v.data(),v.data()+v.size());
            // middle element
            size_t n = x.size()/2;
            // sort nth element of array
            nth_element(x.begin(),x.begin()+n,x.end());
            // if evenly sized, return average of middle two elements
            if (x.size() % 2 == 0) {
                nth_element(x.begin(),x.begin()+n-1,x.end());
                return (x.at(n) + x.at(n-1)) / 2;
            }
            // otherwise return middle element
            else
                return x.at(n);
        }
        
        /// returns the (first) index of the element with the middlest value in v
        int argmiddle(vector<float>& v)
        {
            // instantiate a vector
            vector<float> x = v; 
            // middle iterator
            std::vector<float>::iterator middle = x.begin() + x.size()/2;
            // sort nth element of array
            nth_element(x.begin(), middle, x.end());
            // find position of middle value in original array
            std::vector<float>::iterator it = std::find(v.begin(), v.end(), *middle);

            std::vector<float>::size_type pos = std::distance(v.begin(), it);
            /* cout << "middle index: " << pos << "\n"; */
            /* cout << "middle value: " << *it << "\n"; */
            return pos;
        }
        
        /// calculate variance when mean provided
        float variance(const ArrayXf& v, float mean) 
        {
            ArrayXf tmp = mean*ArrayXf::Ones(v.size());
            return pow((v - tmp), 2).mean();
        }
        
        /// calculate variance
        float variance(const ArrayXf& v) 
        {
            float mean = v.mean();
            return variance(v, mean);
        }
        
        /// calculate skew
        float skew(const ArrayXf& v) 
        {
            float mean = v.mean();
            ArrayXf tmp = mean*ArrayXf::Ones(v.size());
            
            float thirdMoment = pow((v - tmp), 3).mean();
            float variance = pow((v - tmp), 2).mean();
            
            return thirdMoment/sqrt(pow(variance, 3));
        }
        
        /// calculate kurtosis
        float kurtosis(const ArrayXf& v) 
        {
            float mean = v.mean();
            ArrayXf tmp = mean*ArrayXf::Ones(v.size());
            
            float fourthMoment = pow((v - tmp), 4).mean();
            float variance = pow((v - tmp), 2).mean();
            
            return fourthMoment/pow(variance, 2);
        }
        
        float covariance(const ArrayXf& x, const ArrayXf& y)
        {
            float meanX = x.mean();
            float meanY = y.mean();
            //float count = x.size();
            
            ArrayXf tmp1 = meanX*ArrayXf::Ones(x.size());
            ArrayXf tmp2 = meanY*ArrayXf::Ones(y.size());
            
            return ((x - tmp1)*(y - tmp2)).mean();
            
        }
        
        float slope(const ArrayXf& x, const ArrayXf& y)
            // y: rise dimension, x: run dimension. slope = rise/run
        {
            return covariance(x, y)/variance(x);
        }

        // Pearson correlation    
        float pearson_correlation(const ArrayXf& x, const ArrayXf& y)
        {
            return pow(covariance(x,y),2) / (variance(x) * variance(y));
        }
        /// median absolute deviation
        float mad(const ArrayXf& x) 
        {
            // returns median absolute deviation (MAD)
            // get median of x
            float x_median = median(x);
            //calculate absolute deviation from median
            ArrayXf dev(x.size());
            for (int i =0; i < x.size(); ++i)
                dev(i) = fabs(x(i) - x_median);
            // return median of the absolute deviation
            return median(dev);
        }

	    Timer::Timer(bool run)
	    {
		    if (run)
        		Reset();
	    }
	    void Timer::Reset()
	    {
		    _start = high_resolution_clock::now();
	    }
        std::chrono::duration<float> Timer::Elapsed() const
	    {
		    return high_resolution_clock::now() - _start;
	    }
       
        /// fit the scale and offset of data. 
        void Normalizer::fit(MatrixXf& X, const vector<char>& dt)
        {
            scale.clear();
            offset.clear();
            dtypes = dt; 
            for (unsigned int i=0; i<X.rows(); ++i)
            {
                // mean center
                VectorXf tmp = X.row(i).array()-X.row(i).mean();
                // scale by the standard deviation
                scale.push_back(std::sqrt((tmp.array()).square().sum()/(tmp.size()-1)));
                offset.push_back(X.row(i).mean());
            }
              
        }
      
        /// normalize matrix.
        void Normalizer::normalize(MatrixXf& X)
        {  
            // normalize features
            for (unsigned int i=0; i<X.rows(); ++i)
            {
                if (std::isinf(scale.at(i)))
                {
                    X.row(i) = VectorXf::Zero(X.row(i).size());
                    continue;
                }
                // scale, potentially skipping binary and categorical rows
                if (this->scale_all || dtypes.at(i)=='f')                   
                {
                    X.row(i) = X.row(i).array() - offset.at(i);
                    if (scale.at(i) > NEAR_ZERO)
                        X.row(i) = X.row(i).array()/scale.at(i);
                }
            }
        }
        
        void Normalizer::fit_normalize(MatrixXf& X, const vector<char>& dtypes)
        {
            fit(X, dtypes);
            normalize(X);
        }

        /// returns true for elements of x that are infinite
        ArrayXb isinf(const ArrayXf& x)
        {
            ArrayXb infs(x.size());
            for (unsigned i =0; i < infs.size(); ++i)
                infs(i) = std::isinf(x(i));
            return infs;
        }
        
        /// returns true for elements of x that are NaN
        ArrayXb isnan(const ArrayXf& x)
        {
            ArrayXb nans(x.size());
            for (unsigned i =0; i < nans.size(); ++i)
                nans(i) = std::isnan(x(i));
            return nans;

        }
	
        /* Defined in utils.h
        ///template function to convert objects to string for logging
        template <typename T>
        string to_string(const T& value)
        {
            std::stringstream ss;
            ss << value;
            return ss.str();
        }*/
        /// returns the condition number of a matrix.
        float condition_number(const MatrixXf& X)
        {
            /* cout << "X (" << X.rows() << "x" << X.cols() << "): " << X.transpose() << "\n"; */
            /* MatrixXf Y = X; */
            /* try */
            /* { */
            /* JacobiSVD<MatrixXf> svd(Y); */
            BDCSVD<MatrixXf> svd(X);
            /* cout << "JacobiSVD declared\n"; */
            float cond=MAX_FLT; 
            /* cout << "running svals\n"; */
            ArrayXf svals = svd.singularValues();
            /* cout << "svals: " << svals.transpose() << "\n"; */
            if (svals.size()>0)
            {
                cond= svals(0) / svals(svals.size()-1);
            }
            /* cout << "CN: " + std::to_string(cond) + "\n"; */
            return cond;

            /* } */
            /* catch (...) */
            /* { */
            return MAX_FLT;
            /* } */
        }

        /// returns the pearson correlation coefficients of matrix.
        MatrixXf corrcoef(const MatrixXf& X)
        { 
            MatrixXf centered = X.colwise() - X.rowwise().mean();

            /* std::cout << "centered: " << centered.rows() << "x" << centered.cols() << ": " */ 
            /*           << centered << "\n\n"; */
            MatrixXf cov = ( centered * centered.adjoint()) / float(X.cols() - 1);
            /* std::cout << "cov: " << cov.rows() << "x" << cov.cols() << ": " << cov << "\n\n"; */
            VectorXf tmp = 1/cov.diagonal().array().sqrt();
            auto d = tmp.asDiagonal();
            /* std::cout << "1/sqrt(diag(cov)): " << d.rows() << "x" << d.cols() << ": " */ 
            /*           << d.diagonal() << "\n"; */
            MatrixXf corrcoef = d * cov * d;
            /* std::cout << "cov/d: " << corrcoef.rows() << "x" << corrcoef.cols() << ": " */ 
            /*           << corrcoef << "\n"; */
            return corrcoef;
        }

        // returns the mean of the pairwise correlations of a matrix.
        float mean_square_corrcoef(const MatrixXf& X)
        {
            MatrixXf tmp = corrcoef(X).triangularView<StrictlyUpper>();
            float N = tmp.rows()*(tmp.rows()-1)/2;
            /* cout << "triangular strictly upper view: " << tmp << "\n"; */
            return tmp.array().square().sum()/N;
        }
        
        void Log_Stats::update(int index,
                               float timer_count,
                               float bst_score,
                               float bst_score_v,
                               float md_score,
                               float md_loss_v,
                               unsigned md_size,
                               unsigned md_complexity,
                               unsigned md_num_params,
                               unsigned md_dim)
        {
            generation.push_back(index+1);
            time.push_back(timer_count);
            best_score.push_back(bst_score);
            best_score_v.push_back(bst_score_v);
            med_score.push_back(md_score);
            med_loss_v.push_back(md_loss_v);
            med_size.push_back(md_size);
            med_complexity.push_back(md_complexity);
            med_num_params.push_back(md_num_params);
            med_dim.push_back(md_dim);
        }
        
    }
 
} 
