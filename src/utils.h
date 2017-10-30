


#include <Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;

namespace FT{
 
    /*!
     * load csv file into matrix. 
     */
    template<typename M>
    M load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, 
                                RowMajor>>(values.data(), rows, values.size()/rows);
    }
    
    /*!
     * check if element is in vector.
     */
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
    
    /*!
     * check if element is not in vector.
     */
    template<typename T>
    bool not_in(const vector<T>& v, const T& i )
    {
        /* true if i is in v, else false. */
        for (const auto& el : v)
        {
            if (i == el)
                return false;
        }
        return true;
    }

    /*!
     * median 
     */
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

    //! median absolute deviation
    double mad(const ArrayXd& x) 
    {
        // returns median absolute deviation (MAD)
        // get median of x
        double x_median = median(x);
        //calculate absolute deviation from median
        ArrayXd dev(x.size());
        for (int i =0; i < x.size(); ++i)
            dev(i) = abs(x(i) - x_median);
        // return median of the absolute deviation
        return median(dev);
    }

	template <typename T>
	vector<size_t> argsort(const vector<T> &v) {

		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

		return idx;
	}
} 
