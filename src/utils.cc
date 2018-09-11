/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "utils.h"
#include "rnd.h"

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
    
    /// load csv file into matrix. 
    void load_csv (const std::string & path, MatrixXd& X, VectorXd& y, vector<string>& names, 
                   vector<char> &dtypes, bool& binary_endpoint, char sep) 
    {
        std::ifstream indata;
        indata.open(path);
        if (!indata.good())
            HANDLE_ERROR_THROW("Invalid input file " + path + "\n"); 
            
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
       
        dtypes = find_dtypes(X);

        cout << "dtypes: " ; 
        for (unsigned i = 0; i < dtypes.size(); ++i) 
        {
            cout << names.at(i) << " : " << dtypes.at(i);
            cout << "\n";
        }


/*         // get feature types (binary or continuous/categorical) */
/*         int i, j; */
/*         bool isBinary; */
/*         bool isCategorical; */
/*         std::map<double, bool> uniqueMap; */
/*         for(i = 0; i < X.rows(); i++) */
/*         { */
/*             isBinary = true; */
/*             isCategorical = true; */
/*             uniqueMap.clear(); */
            
/*             for(j = 0; j < X.cols(); j++) */
/*             { */
/*                 if(X(i, j) != 0 && X(i, j) != 1) */
/*                     isBinary = false; */
/*                 if(X(i,j) != floor(X(i, j)) && X(i,j) != ceil(X(i,j))) */
/*                     isCategorical = false; */
/*                 else */
/*                     uniqueMap[X(i, j)] = true; */
/*             } */
        
/*             if(isBinary) */
/*                 dtypes.push_back('b'); */
/*             else */
/*             { */
/*                 if(isCategorical && uniqueMap.size() < 10) */
/*                     dtypes.push_back('c'); */    
/*                 else */
/*                     dtypes.push_back('f'); */
/*             } */
/*         } */
        
        // check if endpoint is binary
        binary_endpoint = (y.array() == 0 || y.array() == 1).all();
        
    }
    
    /*!
     * load longitudinal csv file into matrix. 
     */
    void load_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep)
    {
        std::map<string, std::map<int, std::pair<std::vector<double>, std::vector<double> > > > dataMap;
        std::ifstream indata;
        indata.open(path);
        if (!indata.good())
            HANDLE_ERROR_THROW("Invalid input file " + path + "\n"); 
            
        std::string line, firstKey = "";
       
        string header;
        std::getline(indata, header); 
    
        std::stringstream lineStream(header);
        
        std::map<string,int> head_to_col;
        for (int i = 0; i<4; ++i)
        {
            string tmp; 
            std::getline(lineStream,tmp, sep);
            head_to_col[tmp] = i;
        }
        
        while (std::getline(indata, line)) 
        {
            std::stringstream lineStream(line);
            std::string sampleNo, value, time, type;
            
            vector<string> cols(4); 
            std::getline(lineStream, cols.at(0), sep);
            std::getline(lineStream, cols.at(1), sep);
            std::getline(lineStream, cols.at(2), sep);
            std::getline(lineStream, cols.at(3), sep);
           
            sampleNo = cols.at(head_to_col["id"]);
            time = cols.at(head_to_col["date"]);
            value = cols.at(head_to_col["value"]);
            type = cols.at(head_to_col["name"]);

            type = trim(type);
            
            if(!firstKey.compare(""))
                firstKey = type;
            /* cout << "sampleNo: " << sampleNo << ", time: " << time << ", value: " << value */ 
                 /* << ", type: " << type << "\n"; */
            dataMap[type][std::stoi(sampleNo)].first.push_back(std::stod(value));
            dataMap[type][std::stoi(sampleNo)].second.push_back(std::stod(time));
        }
        
        int numTypes = dataMap.size();
        int numSamples = dataMap[firstKey].size();
        int x;
        
        for ( const auto &val: dataMap )
        {
            for(x = 0; x < numSamples; ++x)
            {
                ArrayXd arr1 = Map<ArrayXd>(dataMap[val.first][x].first.data(), dataMap[val.first][x].first.size());
                ArrayXd arr2 = Map<ArrayXd>(dataMap[val.first][x].second.data(), dataMap[val.first][x].second.size());
                Z[val.first].first.push_back(arr1);
                Z[val.first].second.push_back(arr2);

            }
            
        }
    }
    
    /*!
     * load partial longitudinal csv file into matrix according to idx vector
     */
    void load_partial_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep, vector<int> idx)
    {
        /* loads data from the longitudinal file, with idx providing the id numbers of each row in
         * the main data (X and y).
         * I.e., idx[k] = the id of samples in Z associated with sample k in X and y
         */
        std::map<int, bool> idMap;
        std::map<int, int> idLoc;
        unsigned i = 0;
        for(const auto& id : idx){
            idMap[id] = true;
            idLoc[id] = i;
            ++i;
        }
        std::map<string, std::map<int, std::pair<std::vector<double>, std::vector<double> > > > dataMap;
        std::ifstream indata;
        indata.open(path);
        if (!indata.good())
            HANDLE_ERROR_THROW("Invalid input file " + path + "\n");
        
        std::string line, firstKey = "";
       
        // get header
        string header;
        std::getline(indata, header); 
    
        std::stringstream lineStream(header);
        
        std::map<string,int> head_to_col;
        for (int i = 0; i<4; ++i)
        {
            string tmp; 
            std::getline(lineStream,tmp, sep);
            head_to_col[tmp] = i;
        }
        
        while (std::getline(indata, line)) 
        {
            std::stringstream lineStream(line);
            std::string sampleNo, value, time, type;
            
            vector<string> cols(4); 
            std::getline(lineStream, cols.at(0), sep);
            std::getline(lineStream, cols.at(1), sep);
            std::getline(lineStream, cols.at(2), sep);
            std::getline(lineStream, cols.at(3), sep);
           
            sampleNo = cols.at(head_to_col["id"]);
            time = cols.at(head_to_col["date"]);
            value = cols.at(head_to_col["value"]);
            type = cols.at(head_to_col["name"]);

            type = trim(type);
            
            if(!firstKey.compare(""))
                firstKey = type;
            
            int sNo = std::stoi(sampleNo);
            if(idMap.find(sNo) != idMap.end())
            {
                if(idMap[sNo] == true)
                {
                    dataMap[type][idLoc[sNo]].first.push_back(std::stod(value));
                    dataMap[type][idLoc[sNo]].second.push_back(std::stod(time));
                    ++i;
                }
            }
        }
        
        int numSamples = dataMap[firstKey].size();
        int numTypes = dataMap.size();	
        
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


    /*
    /// check if element is in vector.
    template<typename T>
    bool in(const vector<T> v, const T& i)
    {
        // true if i is in v, else false.
        for (const auto& el : v)
        {
            if (i == el)
                return true;
        }
        return false;
    }*/
   
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
	
	/*template <typename T, typename Traits>
	std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, 
                                                     const Timer& timer)
	{
		return out << timer.Elapsed().count();
	}*/
 
    /*
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
    }*/
    
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
    

	
	/*
    /// returns unique elements in vector
    template <typename T>
    vector<T> unique(vector<T> w)   // note intentional copy
    {
        std::sort(w.begin(),w.end());
        typename vector<T>::iterator it;
        it = std::unique(w.begin(),w.end());
        w.resize(std::distance(w.begin(), it));
        return w;
    }*/
    
    /*
    /// returns unique elements in Eigen vector
    template <typename T>
    vector<T> unique(Matrix<T, Dynamic, 1> w)   // note intentional copy
    {
        vector<T> wv( w.data(), w.data()+w.rows());
        return unique(wv);
    }*/

    void printProgress (double percentage)
    {
        int val = (int) (percentage * 100);
        int lpad = (int) (percentage * PBWIDTH);
        int rpad = PBWIDTH - lpad;
        printf ("\rCompleted %3d%% [%.*s%*s]", val, lpad, PBSTR.c_str(), rpad, "");
        fflush (stdout);
        if(val == 100)
            cout << "\n";
    }
    
    /*
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
        JacobiSVD<MatrixXd> svd(X);
        double cond = svd.singularValues()(0) 
                / svd.singularValues()(svd.singularValues().size()-1);
        return cond;
    }
    /// returns the pearson correlation coefficients of matrix.
    MatrixXd corrcoef(const MatrixXd& X)
    { 
        MatrixXd centered = X.colwise() - X.rowwise().mean();

        std::cout << "centered: " << centered.rows() << "x" << centered.cols() << ": " 
                  << centered << "\n\n";
        MatrixXd cov = ( centered * centered.adjoint()) / double(X.cols() - 1);
        std::cout << "cov: " << cov.rows() << "x" << cov.cols() << ": " << cov << "\n\n";
        VectorXd tmp = 1/cov.diagonal().array().sqrt();
        auto d = tmp.asDiagonal();
        std::cout << "1/sqrt(diag(cov)): " << d.rows() << "x" << d.cols() << ": " 
                  << d.diagonal() << "\n";
        MatrixXd corrcoef = d * cov * d;
        std::cout << "cov/d: " << corrcoef.rows() << "x" << corrcoef.cols() << ": " 
                  << corrcoef << "\n";
        return corrcoef;
    }
    // returns the mean of the pairwise correlations of a matrix.
    double mean_square_corrcoef(const MatrixXd& X)
    {
        MatrixXd tmp = corrcoef(X).triangularView<StrictlyUpper>();
        double N = tmp.rows()*(tmp.rows()-1)/2;
        cout << "triangular strictly upper view: " << tmp << "\n";
        return tmp.array().square().sum()/N;
    }
 
} 
