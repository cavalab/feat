/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ostream>
#include <map>
#include "../init.h"
#include "error.h"
//#include "data.h"

using namespace Eigen;

namespace FT{
/**
 * @namespace FT::Util
 * @brief namespace containing various utility functions used in Feat
 */
namespace Util{
     
extern string PBSTR;

extern int PBWIDTH;

/// limits node output to be between MIN_FLT and MAX_FLT
void clean(ArrayXf& x);
void clean(VectorXf& x);

std::string ltrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
 
std::string rtrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
 
std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r ");

/// check if element is in vector.
template<typename T>
bool in(const vector<T> v, const T& i)
{
    return std::find(v.begin(), v.end(), i) != v.end();
}

/// calculate median
float median(const ArrayXf& v);

/// calculate variance when mean provided
float variance(const ArrayXf& v, float mean);

/// calculate variance
float variance(const ArrayXf& v);

/// calculate skew
float skew(const ArrayXf& v);

/// calculate kurtosis
float kurtosis(const ArrayXf& v);

/// covariance of x and y
float covariance(const ArrayXf& x, const ArrayXf& y);

/// slope of x/y
float slope(const ArrayXf& x, const ArrayXf& y);

/// the normalized covariance of x and y
float pearson_correlation(const ArrayXf& x, const ArrayXf& y);

/// median absolute deviation
float mad(const ArrayXf& x);

/// return indices that sort a vector
template <typename T>
vector<size_t> argsort(const vector<T> &v, bool ascending=true) 
{
    // initialize original index locations
    vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    if (ascending)
    {
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    }
    else
    {
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    }

    return idx;
}

/// class for timing things.
class Timer 
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;

    typedef std::chrono::seconds seconds;

    public:
        explicit Timer(bool run = false);
    
        void Reset();
    
        std::chrono::duration<float> Elapsed() const;
        
        template <typename T, typename Traits>
        friend std::basic_ostream<T, Traits>& operator<<(
                std::basic_ostream<T, Traits>& out, const Timer& timer)
        {
            return out << timer.Elapsed().count();
        }
                                                         
        private:
            high_resolution_clock::time_point _start;
    
};

/// return the softmax transformation of a vector.
template <typename T>
vector<T> softmax(const vector<T>& w)
{
    int x;
    T sum = 0;
    vector<T> w_new;
    
    for(x = 0; x < w.size(); ++x)
        sum += exp(w[x]);
        
    for(x = 0; x < w.size(); ++x)
        w_new.push_back(exp(w[x])/sum);
        
    return w_new;
}

/// normalizes a matrix to unit variance, 0 mean centered.
struct Normalizer
{
    Normalizer(bool sa=true): scale_all(sa) {};
    vector<float> scale;
    vector<float> offset;
    vector<char> dtypes;
    bool scale_all;
    
    /// fit the scale and offset of data. 
    template <typename T> 
    void fit(const MatrixBase<T>& X, const vector<char>& dt)
    {
        scale.clear();
        offset.clear();
        dtypes = dt; 
        for (unsigned int i=0; i<X.rows(); ++i)
        {
            // mean center
            auto tmp = X.row(i).array()-X.row(i).mean();
            /* VectorXf tmp; */
            // scale by the standard deviation
            scale.push_back(
                std::sqrt(tmp.square().sum()/(tmp.size()-1)));
            offset.push_back(float(X.row(i).mean()));
        }
          
    }
    /// normalize matrix.
    template <typename T> 
    void normalize(MatrixBase<T>& X)
    {  
        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i)
        {
            if (std::isinf(scale.at(i)))
            {
                /* X.row(i) = Matrix<T, Dynamic, 1>::Zero(X.row(i).size()); */
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
    /// fit then normalize
    template <typename T> 
    void fit_normalize(MatrixBase<T>& X, 
            const vector<char>& dtypes) 
    {
        this->fit(X, dtypes);
        this->normalize(X);
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Normalizer, scale, offset, dtypes, scale_all)


/// returns true for elements of x that are infinite
ArrayXb isinf(const ArrayXf& x);

/// returns true for elements of x that are NaN
ArrayXb isnan(const ArrayXf& x);

/// calculates data types for each column of X
vector<char> find_dtypes(const MatrixXf &X);

/// returns unique elements in vector
template <typename T>
vector<T> unique(vector<T> w)
{
    std::sort(w.begin(),w.end());
    typename vector<T>::iterator it;
    it = std::unique(w.begin(),w.end());
    w.resize(std::distance(w.begin(), it));
    return w;
}

/// returns unique elements in Eigen matrix of variable rows/cols
template <typename T>
vector<T> unique(Matrix<T, -1, -1> w)
{
    vector<T> wv( w.data(), w.data()+w.size());
    return unique(wv);
}

/// returns unique elements in Eigen vector
template <typename T>
vector<T> unique(Matrix<T, -1, 1> w)
{
    vector<T> wv( w.data(), w.data()+w.size());
    return unique(wv);
}

/// returns unique elements in 1d Eigen array
template <typename T>
vector<T> unique(Array<T, -1, 1> w)
{
    vector<T> wv( w.data(), w.data()+w.rows()*w.cols());
    return unique(wv);
}
 
/// returns the condition number of a matrix.
float condition_number(const MatrixXf& X);
  
/// returns the pearson correlation coefficients of matrix.
MatrixXf corrcoef(const MatrixXf& X);

// returns the mean of the pairwise correlations of a matrix.
float mean_square_corrcoef(const MatrixXf& X);

/// returns the (first) index of the element with the middlest value in v
int argmiddle(vector<float>& v);

struct Log_Stats
{
    vector<int> generation;
    vector<float> time;
    vector<float> min_loss;
    vector<float> min_loss_v;
    vector<float> med_loss;
    vector<float> med_loss_v;
    vector<unsigned> med_size;
    vector<unsigned> med_complexity;
    vector<unsigned> med_num_params;
    vector<unsigned> med_dim;
    
    void update(int index,
                float timer_count,
                float bst_score,
                float bst_score_v,
                float md_score,
                float md_loss_v,
                unsigned md_size,
                unsigned md_complexity,
                unsigned md_num_params,
                unsigned md_dim);
};

typedef struct Log_Stats Log_stats;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Log_Stats,
    generation,
    time,
    min_loss,
    min_loss_v,
    med_loss,
    med_loss_v,
    med_size,
    med_complexity,
    med_num_params,
    med_dim);

///template function to convert objects to string for logging
template <typename T>
std::string to_string(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T>
std::string to_string(const T a_value, const int n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

///takes a vector string and returns it as a delimited string.
std::string ravel(const vector<string>& v, string sep=",");

} // Util

} // FT
#endif
