/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "utils.h"
#include "rnd.h"
#include <unordered_set>

namespace FT{
 
    string PBSTR = "============================================================================"
                   "========================";
    int PBWIDTH = 80;
 
    /// limits node output to be between MIN_DBL and MAX_DBL
    void clean(ArrayXd& x)
    {
        x = (x < MIN_DBL).select(MIN_DBL,x);
        x = (isinf(x)).select(MAX_DBL,x);
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
    vector<char> find_dtypes(MatrixXd &X)
    {
	    vector<char> dtypes;
	    
	    // get feature types (binary or continuous/categorical)
        int i, j;
        bool isBinary;
        bool isCategorical;
        std::map<double, bool> uniqueMap;
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

        return dtypes;

	}
    
    void reorder_longitudinal(vector<ArrayXd> &vec1, vector<ArrayXd> &vec2,
                             vector<long> const &order)
    {   
    
        for( int s = 1, d; s < order.size(); ++ s )
        {
            for ( d = order[s]; d < s; d = order[d] );
            
            if ( d == s )
            {
                while ( d = order[d], d != s )
                {
                    swap(vec1[s], vec1[d]);
                    swap(vec2[s], vec2[d]);
                }
            }
        }
    }

    /// calculate median
    double median(const ArrayXd& v) 
    {
        // instantiate a vector
        vector<double> x(v.size());
        x.assign(v.data(),v.data()+v.size());
        // middle element
        size_t n = x.size()/2;
        // sort nth element of array
        nth_element(x.begin(),x.begin()+n,x.end());
        // if evenly sized, return average of middle two elements
        if (x.size() % 2 == 0) {
            nth_element(x.begin(),x.begin()+n-1,x.end());
            return (x[n] + x[n-1]) / 2;
        }
        // otherwise return middle element
        else
            return x[n];
    }
    
    /// calculate variance
    double variance(const ArrayXd& v) 
    {
        double mean = v.mean();
        ArrayXd tmp = mean*ArrayXd::Ones(v.size());
        return pow((v - tmp), 2).mean();
    }
    
    /// calculate skew
    double skew(const ArrayXd& v) 
    {
        double mean = v.mean();
        ArrayXd tmp = mean*ArrayXd::Ones(v.size());
        
        double thirdMoment = pow((v - tmp), 3).mean();
        double variance = pow((v - tmp), 2).mean();
        
        return thirdMoment/sqrt(pow(variance, 3));
    }
    
    /// calculate kurtosis
    double kurtosis(const ArrayXd& v) 
    {
        double mean = v.mean();
        ArrayXd tmp = mean*ArrayXd::Ones(v.size());
        
        double fourthMoment = pow((v - tmp), 4).mean();
        double variance = pow((v - tmp), 2).mean();
        
        return fourthMoment/pow(variance, 2);
    }
    
    double covariance(const ArrayXd& x, const ArrayXd& y)
    {
        double meanX = x.mean();
        double meanY = y.mean();
        //double count = x.size();
        
        ArrayXd tmp1 = meanX*ArrayXd::Ones(x.size());
        ArrayXd tmp2 = meanY*ArrayXd::Ones(y.size());
        
        return ((x - tmp1)*(y - tmp2)).mean();
        
    }
    
    double slope(const ArrayXd& x, const ArrayXd& y)
    {
        return covariance(x, y)/variance(x);
    }

    // Pearson correlation    
    double pearson_correlation(const ArrayXd& x, const ArrayXd& y)
    {
        return pow(covariance(x,y),2) / (variance(x) * variance(y));
    }
    /// median absolute deviation
    double mad(const ArrayXd& x) 
    {
        // returns median absolute deviation (MAD)
        // get median of x
        double x_median = median(x);
        //calculate absolute deviation from median
        ArrayXd dev(x.size());
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
    std::chrono::duration<double> Timer::Elapsed() const
	{
		return high_resolution_clock::now() - _start;
	}
   
    /// fit the scale and offset of data. 
    void Normalizer::fit(MatrixXd& X, const vector<char>& dt)
    {
        scale.clear();
        offset.clear();
        dtypes = dt; 
        for (unsigned int i=0; i<X.rows(); ++i)
        {
            VectorXd tmp = X.row(i).array()-X.row(i).mean();
            scale.push_back(tmp.norm());
            offset.push_back(X.row(i).mean());
        }
          
    }
    
    /// normalize matrix.
    void Normalizer::normalize(MatrixXd& X)
    {  
        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i)
        {
            if (std::isinf(scale.at(i)))
            {
                X.row(i) = VectorXd::Zero(X.row(i).size());
                continue;
            }
            if (dtypes.at(i)=='f')   // skip binary and categorical rows
            {
                X.row(i) = X.row(i).array() - offset.at(i);
                if (scale.at(i) > NEAR_ZERO)
                    X.row(i) = X.row(i).array()/scale.at(i);
            }
        }
    }
    
    void Normalizer::fit_normalize(MatrixXd& X, const vector<char>& dtypes)
    {
        fit(X, dtypes);
        normalize(X);
    }

    /// returns true for elements of x that are infinite
    ArrayXb isinf(const ArrayXd& x)
    {
        ArrayXb infs(x.size());
        for (unsigned i =0; i < infs.size(); ++i)
            infs(i) = std::isinf(x(i));
        return infs;
    }
    
    /// returns true for elements of x that are NaN
    ArrayXb isnan(const ArrayXd& x)
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
    double condition_number(const MatrixXd& X)
    {
        /* cout << "X (" << X.rows() << "x" << X.cols() << "): " << X.transpose() << "\n"; */
        /* MatrixXd Y = X; */
        /* try */
        /* { */
        /* JacobiSVD<MatrixXd> svd(Y); */
        BDCSVD<MatrixXd> svd(X);
        /* cout << "JacobiSVD declared\n"; */
        double cond=MAX_DBL; 
        /* cout << "running svals\n"; */
        ArrayXd svals = svd.singularValues();
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
        return MAX_DBL;
        /* } */
    }

    /// returns the pearson correlation coefficients of matrix.
    MatrixXd corrcoef(const MatrixXd& X)
    { 
        MatrixXd centered = X.colwise() - X.rowwise().mean();

        /* std::cout << "centered: " << centered.rows() << "x" << centered.cols() << ": " */ 
        /*           << centered << "\n\n"; */
        MatrixXd cov = ( centered * centered.adjoint()) / double(X.cols() - 1);
        /* std::cout << "cov: " << cov.rows() << "x" << cov.cols() << ": " << cov << "\n\n"; */
        VectorXd tmp = 1/cov.diagonal().array().sqrt();
        auto d = tmp.asDiagonal();
        /* std::cout << "1/sqrt(diag(cov)): " << d.rows() << "x" << d.cols() << ": " */ 
        /*           << d.diagonal() << "\n"; */
        MatrixXd corrcoef = d * cov * d;
        /* std::cout << "cov/d: " << corrcoef.rows() << "x" << corrcoef.cols() << ": " */ 
        /*           << corrcoef << "\n"; */
        return corrcoef;
    }

    // returns the mean of the pairwise correlations of a matrix.
    double mean_square_corrcoef(const MatrixXd& X)
    {
        MatrixXd tmp = corrcoef(X).triangularView<StrictlyUpper>();
        double N = tmp.rows()*(tmp.rows()-1)/2;
        /* cout << "triangular strictly upper view: " << tmp << "\n"; */
        return tmp.array().square().sum()/N;
    }
 
} 
