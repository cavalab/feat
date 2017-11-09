/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>
#include <ostream>

using namespace Eigen;

namespace FT{
 
    /*!
     * load csv file into matrix. 
     */
    
    void load_csv (const std::string & path, MatrixXd& X, VectorXd& y, vector<string> names, 
                   char sep=',') 
    {
        std::ifstream indata;
        indata.open(path);
        if (!indata.good())
        { 
            std::cerr << "Invalid input file " + path + "\n"; 
            exit(1);
        }
        std::string line;
        std::vector<double> values, targets;
        unsigned rows=0, col=0, target_col = 0;
        
        while (std::getline(indata, line)) 
        {
            std::stringstream lineStream(line);
            std::string cell;
            
            while (std::getline(lineStream, cell, sep)) 
            {                
                if (rows==0) // read in header
                {
                    if (!cell.compare("class") || !cell.compare("target") 
                            || !cell.compare("label"))
                        target_col = col;                    
                    else
                        names.push_back(cell);
                }
                else if (col != target_col) 
                    values.push_back(std::stod(cell));
                else
                    targets.push_back(std::stod(cell));
                ++col;
            }
            ++rows;
            col=0;   
        }
        X = Map<MatrixXd>(values.data(), rows-1, values.size()/(rows-1));
        X.transposeInPlace();
        y = Map<VectorXd>(targets.data(), targets.size());
        assert(X.cols() == y.size() && "different numbers of samples in X and y");
        assert(X.rows() == names.size() && "header missing or incorrect number of feature names");
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

    /// median absolute deviation
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

    /// return indices that sort a vector
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

    /// class for timing things.
    class Timer 
	{
        typedef std::chrono::high_resolution_clock high_resolution_clock;
		typedef std::chrono::seconds seconds;
		public:
			explicit Timer(bool run = false)
			{
				if (run)
		    		Reset();
			}
			void Reset()
			{
				_start = high_resolution_clock::now();
			}
            std::chrono::duration<double> Elapsed() const
			{
				return high_resolution_clock::now() - _start;
			}
			template <typename T, typename Traits>
			friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, 
                                                             const Timer& timer)
			{
				return out << timer.Elapsed().count();
			}
			private:
			    high_resolution_clock::time_point _start;
			
    };

    /// split input data into training and validation sets. 
    void train_test_split(MatrixXd& X, VectorXd& y, MatrixXd& X_t, MatrixXd& X_v, VectorXd& y_t, 
                          VectorXd& y_v, bool shuffle)
    {
        /* @params X: n_features x n_samples matrix of training data
         * @params Y: n_samples vector of training labels
         * @params shuffle: whether or not to shuffle X and y
         * @returns X_t, X_v, y_t, y_v: training and validation matrices
         */
        if (shuffle)     // generate shuffle index for the split
        {
            Eigen::PermutationMatrix<Dynamic,Dynamic> perm(X.cols());
            perm.setIdentity();
            r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
            X = X * perm;       // shuffles columns of X
            y = perm * y;       // shuffle y too  
        }
        
        // map training and test sets  
        X_t = MatrixXd::Map(X.data(),X_t.rows(),X_t.cols());
        X_v = MatrixXd::Map(X.data()+X_t.rows()*X_t.cols(),X_v.rows(),X_v.cols());

        y_t = VectorXd::Map(y.data(),y_t.size());
        y_v = VectorXd::Map(y.data()+y_t.size(),y_v.size());

    
    }
} 
