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
 
    double NEAR_ZERO = 0.0000001;

    /*!
     * load csv file into matrix. 
     */
    
    void load_csv (const std::string & path, MatrixXd& X, VectorXd& y, vector<string> names, 
                   vector<char> &dtypes, bool& binary_endpoint, char sep=',') 
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
        //cout<<"Rows read are "<<rows<<std::endl;
        //cout<<"Data size is "<<values.size()<<std::endl;
        
        /*cout<<"Values are\n";
        int x;
        for(x = 0; x < values.size(); x++)
            cout<<values[x]<<std::endl;
        */
        
        //X = Map<MatrixXd>(values.data(), rows-1, values.size()/(rows-1));
        X = Map<MatrixXd>(values.data(), values.size()/(rows-1), rows-1);
        //cout<<"X is\n"<<X<<std::endl;
        //X.transposeInPlace();
        y = Map<VectorXd>(targets.data(), targets.size());
        assert(X.cols() == y.size() && "different numbers of samples in X and y");
        assert(X.rows() == names.size() && "header missing or incorrect number of feature names");
        
        // get feature types (binary or continuous/categorical)
        int i, j;
        bool isBinary;
        for(i = 0; i < X.rows(); i++)
        {
            isBinary = true;
            //cout<<"Checking for column "<<i<<std::endl;
            for(j = 0; j < X.cols(); j++)
            {
                //cout<<"Value is "<<X(i, j)<<std::endl;
                if(X(i, j) != 0 && X(i, j) != 1)
                    isBinary = false;
            }
            if(isBinary)
                dtypes.push_back('b');
            else
                dtypes.push_back('f');
        }
        // check if endpoint is binary
        binary_endpoint = (y.array() == 0 || y.array() == 1).all();
        
       // cout<<"X^T is\n";
       // for (unsigned i=0; i< dtypes.size(); ++i)
       //     cout << names[i] << "[" << dtypes[i] << "] ";
       // cout << "\n" << X.transpose()<<std::endl;
       // cout<<"Y is\n"<<y<<std::endl;

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
            dev(i) = fabs(x(i) - x_median);
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
         * @params y: n_samples vector of training labels
         * @params shuffle: whether or not to shuffle X and y
         * @returns X_t, X_v, y_t, y_v: training and validation matrices
         */
        if (shuffle)     // generate shuffle index for the split
        {
            Eigen::PermutationMatrix<Dynamic,Dynamic> perm(X.cols());
            perm.setIdentity();
            r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
            //std::cout << "permutation matrix: " << perm << "\n";
            X = X * perm;       // shuffles columns of X
            y = (y.transpose() * perm).transpose() ;       // shuffle y too  
        }
        
        // map training and test sets  
        X_t = MatrixXd::Map(X.data(),X_t.rows(),X_t.cols());
        X_v = MatrixXd::Map(X.data()+X_t.rows()*X_t.cols(),X_v.rows(),X_v.cols());

        y_t = VectorXd::Map(y.data(),y_t.size());
        y_v = VectorXd::Map(y.data()+y_t.size(),y_v.size());

    
    }

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
    
    /// normalize matrix.
    void normalize(MatrixXd& X, const vector<char>& dtypes)
    {   
        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i){
            if (std::isinf(X.row(i).norm()))
            {
                X.row(i) = VectorXd::Zero(X.row(i).size());
                continue;
            }
            if (dtypes.at(i)!='b')   // skip binary rows
            {
                X.row(i) = X.row(i).array() - X.row(i).mean();
                if (X.row(i).norm() > NEAR_ZERO)
                    X.row(i).normalize();
            }
        }
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

    /// returns unique elements in vector
    template <typename T>
    vector<T> unique(vector<T> w)   // note intentional copy
    {
        std::sort(w.begin(),w.end());
        typename vector<T>::iterator it;
        it = std::unique(w.begin(),w.end());
        w.resize(std::distance(w.begin(), it));
        return w;
    }
    /// returns unique elements in Eigen vector
    template <typename T>
    vector<T> unique(Matrix<T, Dynamic, 1> w)   // note intentional copy
    {
        vector<T> wv( w.data(), w.data()+w.rows());
        return unique(wv);
    }
} 
