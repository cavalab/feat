/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>
#include <ostream>
#include <map>
#include "../init.h"
#include "../util/error.h"
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
			    friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, 
                                                                 const Timer& timer)
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
            vector<float> scale;
            vector<float> offset;
            vector<char> dtypes;

            /// fit the scale and offset of data. 
            void fit(MatrixXf& X, const vector<char>& dt);
            
            /// normalize matrix.
            void normalize(MatrixXf& X);
            
            void fit_normalize(MatrixXf& X, const vector<char>& dtypes);
        };

        /// returns true for elements of x that are infinite
        ArrayXb isinf(const ArrayXf& x);
        
        /// returns true for elements of x that are NaN
        ArrayXb isnan(const ArrayXf& x);
       
        /// calculates data types for each column of X
        vector<char> find_dtypes(MatrixXf &X);
	
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
        
        /// returns unique elements in Eigen vector
        template <typename T>
        vector<T> unique(Matrix<T, Dynamic, 1> w)
        {
            vector<T> wv( w.data(), w.data()+w.rows());
            return unique(wv);
        }
        
        ///template function to convert objects to string for logging
        template <typename T>
        string to_string(const T& value)
        {
            std::stringstream ss;
            ss << value;
            return ss.str();
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
            vector<float> best_score;
            vector<float> best_score_v;
            vector<float> med_score;
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

    }

} 
#endif
