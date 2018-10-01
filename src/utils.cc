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
    
    /// load longitudinal csv file into matrix. 
    void load_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep)
    {
        cout << "in load_longitudinal\n";
        std::map<string, std::map<int, std::pair<vector<double>, vector<double> > > > dataMap;
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
        
        int numVars = dataMap.size();
        int numSamples = dataMap[firstKey].size();
        int x;
        
        for ( const auto &val: dataMap )
        {
            for(x = 0; x < numSamples; ++x)
            {
                ArrayXd arr1 = Map<ArrayXd>(dataMap[val.first].at(x).first.data(), 
                                            dataMap[val.first].at(x).first.size());
                ArrayXd arr2 = Map<ArrayXd>(dataMap[val.first].at(x).second.data(), 
                                            dataMap[val.first].at(x).second.size());
                Z[val.first].first.push_back(arr1);
                Z[val.first].second.push_back(arr2);

            }
            
        }

        cout << "exiting load_longitudinal\n";
    }
    
    /*!
     * load partial longitudinal csv file into matrix according to idx vector
     */
    void load_partial_longitudinal(const std::string & path,
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                           char sep, vector<long> idx)
    {
        /* loads data from the longitudinal file, with idx providing the id numbers of each row in
         * the main data (X and y).
         * I.e., idx[k] = the id of samples in Z associated with sample k in X and y
         */
        cout << "in load_partial_longitudinal\n";
        cout << idx.size() << " indices\n";

        std::unordered_set<long> idSet; //(idx.begin(), idx.end());
        idSet.insert(idx.begin(), idx.end());
        // write out info about unordered_set
        cout << "idSet size: " << idSet.size() << "\n";
        cout << "max_size = " << idSet.max_size() << "\n"; 
        cout << "max_bucket_count = " << idSet.max_bucket_count() << "\n";
        cout << "max_load_factor = " << idSet.max_load_factor() << "\n";
        cout << "idSet: ";
        for (auto s : idSet)
            cout << s << ",";
        cout << "\n";
        std::map<int, int> idLoc;
        std::map<int, int> locID;
        unsigned i = 0;
        for(const auto& id : idx){
            /* auto tmp = idSet.insert(id); */
            /* if (!tmp.second || *tmp.first != id) */
            /*     cout << "insert failed on i = " << i << ", id = " << id << "\n"; */
            idLoc[id] = i;
            locID[i] = id;
            ++i;
        }
        /* cout << "\n"; */
        // dataMap maps from the variable name (string) to a map containing 
        // 1) the sample id, and 2) a pair consisting of 
        //      - the variable value (first) and 
        //      - variable date (second)
        std::map<string, std::map<int, std::pair<vector<double>, vector<double> > > > dataMap;
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
            tmp = trim(tmp);
            head_to_col[tmp] = i;
        }
        int nl=0; 
        int nfound=0;
        int nskip=0;
        cout << "reading " << path << "...\n";
        while (std::getline(indata, line)) 
        {
            std::stringstream lineStream(line);
            std::string sampleNo, value, time, name;
            
            vector<string> cols(4); 
            std::getline(lineStream, cols.at(0), sep);
            std::getline(lineStream, cols.at(1), sep);
            std::getline(lineStream, cols.at(2), sep);
            std::getline(lineStream, cols.at(3), sep);
            
            cols.at(3) = trim(cols.at(3));

            sampleNo = cols.at(head_to_col["id"]);
            time = cols.at(head_to_col["date"]);
            value = cols.at(head_to_col["value"]);
            name = cols.at(head_to_col["name"]);

            /* name = trim(name); */
            /* time = trim(time); */
            /* value = trim(value); */
            /* sampleNo = trim(sampleNo); */
            
            /* cout << "sampleNo: " << sampleNo << ", time: " << time << ", value: " << value */ 
            /*      << ", name: " << name << "\n"; */
  
            if(!firstKey.compare(""))
                firstKey = name;
            
            long sNo = std::stol(sampleNo);
            /* if(idSet.find(sNo) != idSet.end())  // if the sample ID is to be included, store it */
            if(in(idx,sNo))  // if the sample ID is to be included, store it
            {
                /* if(idMap.at(sNo) == true)   // WGL: I think this is irrelevant */
                /* { */
                // dataMap[variable-name][sample-id].value=value
                // dataMap[variable-name][sample-id].time=time
                    dataMap[name][idLoc.at(sNo)].first.push_back(std::stod(value));
                    dataMap[name][idLoc.at(sNo)].second.push_back(std::stod(time));
                /* } */
                    ++nfound;
            }
            /* else if (sNo == 552570) */
            /* /1* else if (sNo > 33700) *1/ */
            /* { */
            /*     if (in(idx,sNo)) */
            /*     { */
            /*         cout << sNo << " is in idx, but not found in idSet\n"; */
            /*         cout << "file line: "; */
            /*         for (auto c : cols) */
            /*             cout << c << ","; */
            /*         cout << "\n"; */
            /*     } */
            /* } */
            else
            {
                ++nskip;
            }
            ++nl;
        }
        cout << "read " << nl << " lines of " << path << "\n";
        cout << "stored " << nfound << " lines, skipped " << nskip << "\n";
        // validate dataMap
        // for each dataMap[name], there should be map names from 0 ... numSamples -1

        for ( const auto &val: dataMap )
        {
            bool pass = true;
            int numSamples = val.second.size();
            for (int x = 0; x<numSamples; ++x)
            {
                if (val.second.find(x) == val.second.end())
                {
                    cout << x << " not found (patient id = " << locID[x] << ") in " << val.first 
                         << "\n";
                    pass = false;
                }
            }
            if (!pass) 
                exit(0);
        }
        int numVars = dataMap.size();
        cout << "numVars= " << numVars << "\n";
        
        for ( const auto &val: dataMap )
        {
            cout << "storing " << val.first << "\n";
            int numSamples = val.second.size();
            cout << "numSamples= " << numSamples << "\n";
            cout << "dataMap[val.first].size(): " << dataMap[val.first].size() << "\n"; 
            cout << "x: ";
            for(int x = 0; x < numSamples; ++x)
            {
                cout << x << ",";
                ArrayXd arr1 = Map<ArrayXd>(dataMap[val.first].at(x).first.data(), 
                                            dataMap[val.first].at(x).first.size());
                ArrayXd arr2 = Map<ArrayXd>(dataMap[val.first].at(x).second.data(), 
                                            dataMap[val.first].at(x).second.size());
                Z[val.first].first.push_back(arr1);
                Z[val.first].second.push_back(arr2);
            }
            cout << "\n";
        }
        cout << "Z loaded. contents:\n";
        for (const auto& z : Z)
        {
            cout << "zName: " << z.first << "\n";
            if (z.second.first.size() != z.second.second.size())
            {
                cout << "values and time not the same size\n";
                cout << "values: " << z.second.first.size() << "\n";
                cout << "time: " << z.second.second.size() << "\n";
                exit(0);
            }
            for (unsigned int j = 0; j < z.second.first.size(); ++j)
            {
                cout << "sample " << j << " = " ;
                for (unsigned int k = 0; k < z.second.first.at(j).size(); ++k)
                    cout << z.second.second.at(j)(k) << ":" << z.second.first.at(j)(k) << ",";
                cout << "\n";
            }
            cout << "---\n";
        }
        cout << "exiting load_partial_longitudinal\n";
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
