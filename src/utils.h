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
#include "init.h"
#include "error.h"
//#include "data.h"

using namespace Eigen;

namespace FT{
 
    extern string PBSTR;
    
    extern int PBWIDTH;
 
    /// limits node output to be between MIN_DBL and MAX_DBL
    void clean(ArrayXd& x);

    std::string ltrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
     
    std::string rtrim(std::string str, const std::string& chars = "\t\n\v\f\r ");
     
    std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r ");

    /*!
     * load csv file into matrix. 
     */
    
    void load_csv (const std::string & path, MatrixXd& X, VectorXd& y, vector<string>& names, 
                   vector<char> &dtypes, bool& binary_endpoint, char sep=',');
    
    /*!
     * load longitudinal csv file into matrix. 
     */
    void load_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep=',');
    
    /*!
     * load partial longitudinal csv file into matrix according to idx vector
     */
    void load_partial_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep, vector<int> idx);

    void reorder_longitudinal(vector<ArrayXd> &vec1, vector<ArrayXd> &vec2,
                             vector<int> const &order) 
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
    /// check if element is in vector.
    template<typename T>
    bool in(const vector<T> v, const T& i)
    {
        /* true if i is in v, else false. */
        for (const auto& el : v)
        {
            if (i == el)
                return true;
        }
        return false;
    }
   
    /// calculate median
    double median(const ArrayXd& v);
    
    /// calculate variance
    double variance(const ArrayXd& v);
    
    /// calculate skew
    double skew(const ArrayXd& v);
    
    /// calculate kurtosis
    double kurtosis(const ArrayXd& v);
    
    double covariance(const ArrayXd& x, const ArrayXd& y);
    
    double slope(const ArrayXd& x, const ArrayXd& y);

    /// median absolute deviation
    double mad(const ArrayXd& x);

    /// return indices that sort a vector
	template <typename T>
	vector<size_t> argsort(const vector<T> &v)
	{

		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

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
			
            std::chrono::duration<double> Elapsed() const;
            
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
        
        for(x = 0; x < w.size(); x++)
            sum += exp(w[x]);
            
        for(x = 0; x < w.size(); x++)
            w_new.push_back(exp(w[x])/sum);
            
        return w_new;
    }
    
    struct Normalizer
    {
        vector<double> scale;
        vector<double> offset;
        vector<char> dtypes;

        /// fit the scale and offset of data. 
        void fit(MatrixXd& X, const vector<char>& dt);
        
        /// normalize matrix.
        void normalize(MatrixXd& X);
        
        void fit_normalize(MatrixXd& X, const vector<char>& dtypes);
        
    };

    /// returns true for elements of x that are infinite
    ArrayXb isinf(const ArrayXd& x);
    
    /// returns true for elements of x that are NaN
    ArrayXb isnan(const ArrayXd& x);
    
    vector<char> find_dtypes(MatrixXd &X);
	
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

    void printProgress (double percentage);
    
    ///template function to convert objects to string for logging
    template <typename T>
    string to_string(const T& value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
    
} 
#endif
