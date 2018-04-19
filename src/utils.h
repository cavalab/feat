/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>
#include <ostream>
#include <map>

using namespace Eigen;

namespace FT{
 
    double NEAR_ZERO = 0.0000001;
    
    std::string ltrim(std::string str, const std::string& chars = "\t\n\v\f\r ")
    {
        str.erase(0, str.find_first_not_of(chars));
        return str;
    }
     
    std::string rtrim(std::string str, const std::string& chars = "\t\n\v\f\r ")
    {
        str.erase(str.find_last_not_of(chars) + 1);
        return str;
    }
     
    std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r ")
    {
        return ltrim(rtrim(str, chars), chars);
    }

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
                cell = trim(cell);
                  
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
        
        X = Map<MatrixXd>(values.data(), values.size()/(rows-1), rows-1);
        y = Map<VectorXd>(targets.data(), targets.size());
        
        assert(X.cols() == y.size() && "different numbers of samples in X and y");
        assert(X.rows() == names.size() && "header missing or incorrect number of feature names");
        
        // get feature types (binary or continuous/categorical)
        int i, j;
        bool isBinary;
        for(i = 0; i < X.rows(); i++)
        {
            isBinary = true;
            for(j = 0; j < X.cols(); j++)
                if(X(i, j) != 0 && X(i, j) != 1)
                    isBinary = false;
        
            if(isBinary)
                dtypes.push_back('b');
            else
                dtypes.push_back('f');
        }
        
        // check if endpoint is binary
        binary_endpoint = (y.array() == 0 || y.array() == 1).all();
        
    }
    
    /*!
     * load longitudinal csv file into matrix. 
     */
    void load_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep=',')
    {
        std::map<string, std::map<int, std::pair<std::vector<double>, std::vector<double> > > > dataMap;
        std::ifstream indata;
        indata.open(path);
        if (!indata.good())
        { 
            std::cerr << "Invalid input file " + path + "\n"; 
            exit(1);
        }
        std::string line, firstKey = "";
        
        while (std::getline(indata, line)) 
        {
            std::stringstream lineStream(line);
            std::string sampleNo, value, time, type;
            
            std::getline(lineStream, sampleNo, sep);
            std::getline(lineStream, value, sep);
            std::getline(lineStream, time, sep);
            std::getline(lineStream, type, sep);
            
            type = trim(type);
            
            if(!firstKey.compare(""))
                firstKey = type;
            
            dataMap[type][std::stoi(sampleNo)].first.push_back(std::stod(value));
            dataMap[type][std::stoi(sampleNo)].second.push_back(std::stod(time));
        }
        
        int numTypes = dataMap.size();
        int numSamples = dataMap[firstKey].size();
        
        int x;
        
        for ( const auto &val: dataMap )
        {
            for(x = 0; x < numSamples; x++)
            {
                ArrayXd arr1 = Map<ArrayXd>(dataMap[val.first][x].first.data(), dataMap[val.first][x].first.size());
                ArrayXd arr2 = Map<ArrayXd>(dataMap[val.first][x].second.data(), dataMap[val.first][x].second.size());
                Z[val.first].first.push_back(arr1);
                Z[val.first].second.push_back(arr2);
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
    
    void reorder_logitudinal(vector<ArrayXd> &vec1,
                             vector<ArrayXd> &vec2,
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
    
    void split_longitudinal(std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                            double split)
    {
    
        int size;
        for ( const auto val: Z )
        {
            size = Z[val.first].first.size();
            break;
        }
        
        int testSize = int(size*split);
        int validateSize = int(size*(1-split));
            
        for ( const auto &val: Z )
        {
            vector<ArrayXd> _Z_t_v, _Z_t_t, _Z_v_v, _Z_v_t;
            _Z_t_v.assign(Z[val.first].first.begin(), Z[val.first].first.begin()+testSize);
            _Z_t_t.assign(Z[val.first].second.begin(), Z[val.first].second.begin()+testSize);
            _Z_v_v.assign(Z[val.first].first.begin()+testSize, Z[val.first].first.begin()+testSize+validateSize);
            _Z_v_t.assign(Z[val.first].second.begin()+testSize, Z[val.first].second.begin()+testSize+validateSize);
            
            Z_t[val.first] = make_pair(_Z_t_v, _Z_t_t);
            Z_v[val.first] = make_pair(_Z_v_v, _Z_v_t);
        }
    }



    /// split input data into training and validation sets. 
    void train_test_split(MatrixXd& X,
                          VectorXd& y,
                          std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                          MatrixXd& X_t,
                          MatrixXd& X_v,
                          VectorXd& y_t, 
                          VectorXd& y_v,
                          std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                          std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                          bool shuffle,
                          double split)
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
            X = X * perm;       // shuffles columns of X
            y = (y.transpose() * perm).transpose() ;       // shuffle y too
            
            if(Z.size() > 0)
            {
                std::vector<int> zidx(y.size());
                std::iota(zidx.begin(), zidx.end(), 0);
                Eigen::VectorXi zw = Map<VectorXi>(zidx.data(), zidx.size());
                zw = (zw.transpose()*perm).transpose();       // shuffle zw too
                zidx.assign(((int*)zw.data()), (((int*)zw.data())+zw.size()));
                
                for(auto &val : Z)
                    reorder_logitudinal(val.second.first, val.second.second, zidx);
            }
            
        }
        
        // map training and test sets  
        X_t = MatrixXd::Map(X.data(),X_t.rows(),X_t.cols());
        X_v = MatrixXd::Map(X.data()+X_t.rows()*X_t.cols(),X_v.rows(),X_v.cols());

        y_t = VectorXd::Map(y.data(),y_t.size());
        y_v = VectorXd::Map(y.data()+y_t.size(),y_v.size());
        
        if(Z.size() > 0)
            split_longitudinal(Z, Z_t, Z_v, split);

    
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
    
    struct Normalizer
    {
        vector<double> scale;
        vector<double> offset;
        vector<char> dtypes;

        /// fit the scale and offset of data. 
        void fit(MatrixXd& X, const vector<char>& dt)
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
        void normalize(MatrixXd& X)
        {  
            // normalize features
            for (unsigned int i=0; i<X.rows(); ++i)
            {
                if (std::isinf(scale.at(i)))
                {
                    X.row(i) = VectorXd::Zero(X.row(i).size());
                    continue;
                }
                if (dtypes.at(i)!='b')   // skip binary rows
                {
                    X.row(i) = X.row(i).array() - offset.at(i);
                    if (scale.at(i) > NEAR_ZERO)
                        X.row(i) = X.row(i).array()/scale.at(i);
                }
            }
        }
        void fit_normalize(MatrixXd& X, const vector<char>& dtypes)
        {
            fit(X, dtypes);
            normalize(X);
        }
    };
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
    
    vector<char> find_dtypes(MatrixXd &X)
    {
    	int i, j;
	    bool isBinary;
	    
	    vector<char> dtypes;
	    
	    for(i = 0; i < X.rows(); i++)
	    {
	        isBinary = true;
	        for(j = 0; j < X.cols(); j++)
	            if(X(i, j) != 0 && X(i, j) != 1)
	                isBinary = false;
	                
	        if(isBinary)
	            dtypes.push_back('b');
	        else
	            dtypes.push_back('f');
	    }
	    
	    return dtypes;
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
