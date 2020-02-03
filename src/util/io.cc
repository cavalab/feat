/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "io.h"
#include "utils.h"
/* #include "rnd.h" */
#include <unordered_set>

namespace FT{
    
    namespace Util{
    
        void printProgress (float percentage)
        {
            int val = (int) (percentage * 100);
            int lpad = (int) (percentage * PBWIDTH);
            int rpad = PBWIDTH - lpad;
            printf ("\rCompleted %3d%% [%.*s%*s]", val, lpad, PBSTR.c_str(), rpad, "");
            fflush (stdout);
            if(val == 100)
                cout << "\n";
        }

        /// load csv file into matrix. 
        void load_csv (const std::string & path, MatrixXf& X, VectorXf& y, 
                vector<string>& names, vector<char> &dtypes, bool& binary_endpoint,
                char sep) 
        {
            std::ifstream indata;
            indata.open(path);
            if (!indata.good())
                HANDLE_ERROR_THROW("Invalid input file " + path + "\n"); 
                
            std::string line;
            std::vector<float> values, targets;
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
            
            X = Map<MatrixXf>(values.data(), values.size()/(rows-1), rows-1);
            y = Map<VectorXf>(targets.data(), targets.size());
            
            assert(X.cols() == y.size() && "different numbers of samples in X and y");
            assert(X.rows() == names.size() && "header missing or incorrect number of feature names");
           
            dtypes = find_dtypes(X);

            cout << "dtypes: " ; 
            for (unsigned i = 0; i < dtypes.size(); ++i) 
                cout << names.at(i) << " (" << dtypes.at(i) << "), ";
            cout << "\n";


    /*         // get feature types (binary or continuous/categorical) */
    /*         int i, j; */
    /*         bool isBinary; */
    /*         bool isCategorical; */
    /*         std::map<float, bool> uniqueMap; */
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
                               std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z,
                               char sep)
        {
            std::map<string, std::map<int, std::pair<vector<float>, vector<float> > > > dataMap;
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
               
                sampleNo = cols.at(head_to_col.at("id"));
                time = cols.at(head_to_col.at("date"));
                value = cols.at(head_to_col.at("value"));
                type = cols.at(head_to_col.at("name"));

                type = trim(type);
                
                if(!firstKey.compare(""))
                    firstKey = type;
                /* cout << "sampleNo: " << sampleNo << ", time: " << time << ", value: " << value */ 
                     /* << ", type: " << type << "\n"; */
                dataMap[type][std::stoi(sampleNo)].first.push_back(std::stod(value));
                dataMap[type][std::stoi(sampleNo)].second.push_back(std::stod(time));
            }
            
            int numVars = dataMap.size();
            int numSamples = dataMap.at(firstKey).size();
            int x;
            
            for ( const auto &val: dataMap )
            {
                for(x = 0; x < numSamples; ++x)
                {
                    ArrayXf arr1 = Map<ArrayXf>(dataMap.at(val.first).at(x).first.data(), 
                                                dataMap.at(val.first).at(x).first.size());
                    ArrayXf arr2 = Map<ArrayXf>(dataMap.at(val.first).at(x).second.data(), 
                                                dataMap.at(val.first).at(x).second.size());
                    Z[val.first].first.push_back(arr1);
                    Z[val.first].second.push_back(arr2);

                }
                
            }

        }
        
        /*!
         * load partial longitudinal csv file into matrix according to idx vector
         */
        void load_partial_longitudinal(const std::string & path,
                               std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z,
                               char sep, const vector<int>& idx)
        {
            /* loads data from the longitudinal file, with idx providing the id numbers of each row in
             * the main data (X and y).
             * I.e., idx[k] = the id of samples in Z associated with sample k in X and y
             */
            /* cout << "in load_partial_longitudinal\n"; */
            /* cout << idx.size() << " indices\n"; */
            /* for (unsigned i = 0; i<idx.size(); ++i) */
            /*     cout << i << "," << idx[i] << "\n"; */
            std::unordered_set<int> idSet; //(idx.begin(), idx.end());
            /* idSet.insert(idx.begin(), idx.end()); */
            // write out info about unordered_set
            
            /* cout << "max_size = " << idSet.max_size() << "\n"; */ 
            /* cout << "max_bucket_count = " << idSet.max_bucket_count() << "\n"; */
            /* cout << "max_load_factor = " << idSet.max_load_factor() << "\n"; */
            /* cout << "idSet: "; */
            /* for (auto s : idSet) */
            /*     cout << s << ","; */
            /* cout << "\n"; */
            std::map<int, vector<int>> idLoc;   // maps IDs to X/y row index (i.e. Loc)
            std::map<int, int> locID;           // maps X/y row indices (i.e. loc) to sample IDs 
            unsigned i = 0;
            for(const auto& id : idx){
                auto tmp = idSet.insert(id);
                if (!tmp.second || *tmp.first != id){
                    if(idSet.find(id) == idSet.end())
                    {
                        cout << "failed to find " << id << " in idSet\n"; 
                        cout << "retrying..\n";
                        int blrg=0;
                        while (blrg<100 && (!tmp.second || *tmp.first != id) )
                        {
                            auto tmp = idSet.insert(id);
                            blrg++;
                        }
                        if (blrg == 100)
                            HANDLE_ERROR_THROW("insert failed on i = " + std::to_string(i) + 
                                               " id = " + std::to_string(id));
                    }
                } 
                idLoc[id].push_back(i);
                locID.at(i) = id;
                ++i;
            }
            /* cout << "idSet size: " << idSet.size() << "\n"; */
            /* cout << "idx size: " << idx.size() << "\n"; */
            /* if (idSet.size() != idx.size()) */
            /* { */
            /*     HANDLE_ERROR_THROW("Sample IDs must be unique"); */ 
            /* } */
            /* cout << "\n"; */
            // dataMap maps from the variable name (string) to a map containing 
            // 1) the sample row index in X/y, and 2) a pair consisting of 
            //      - the variable value (first) and 
            //      - variable date (second)
            std::map<string, std::map<int, std::pair<vector<float>, vector<float> > > > dataMap;
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

                sampleNo = cols.at(head_to_col.at("id"));
                time = cols.at(head_to_col.at("date"));
                value = cols.at(head_to_col.at("value"));
                name = cols.at(head_to_col.at("name"));

                /* name = trim(name); */
                /* time = trim(time); */
                /* value = trim(value); */
                /* sampleNo = trim(sampleNo); */
                
                /* cout << "sampleNo: " << sampleNo << ", time: " << time << ", value: " << value */ 
                /*      << ", name: " << name << "\n"; */
      
                if(!firstKey.compare(""))
                    firstKey = name;
                
                int sID = std::stol(sampleNo);
                if(idSet.find(sID) != idSet.end())  // if the sample ID is to be included, store it
                /* if(in(idx,sNo))  // if the sample ID is to be included, store it */
                {
                    // dataMap[variable-name][row-idx].value=value
                    // dataMap[variable-name][row-idx].time=time
                    for (const auto& loc : idLoc.at(sID))
                    {
                        dataMap[name][loc].first.push_back(std::stod(value));
                        dataMap[name][loc].second.push_back(std::stod(time));
                    }
                    /* } */
                    ++nfound;
                }
                else
                {
                    ++nskip;
                }
                ++nl;
            }
            //cout << "read " << nl << " lines of " << path << "\n";
            //cout << "stored " << nfound << " lines, skipped " << nskip << "\n";
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
                        HANDLE_ERROR_THROW(std::to_string(x) + " not found (patient id = " 
                                           + std::to_string(locID.at(x)) + ") in " + 
                                           val.first);
                        pass = false;
                    }
                }
            }
            int numVars = dataMap.size();
            /* cout << "numVars= " << numVars << "\n"; */
            
            for ( const auto &val: dataMap )
            {
                /* cout << "storing " << val.first << "\n"; */
                int numSamples = val.second.size();
                /* cout << "numSamples= " << numSamples << "\n"; */
                /* cout << "dataMap[val.first].size(): " << dataMap[val.first].size() << "\n"; */ 
                /* cout << "x: "; */
                for(int x = 0; x < numSamples; ++x)
                {
                    /* cout << x << ","; */
                    ArrayXf arr1 = Map<ArrayXf>(dataMap.at(val.first).at(x).first.data(), 
                                                dataMap.at(val.first).at(x).first.size());
                    ArrayXf arr2 = Map<ArrayXf>(dataMap.at(val.first).at(x).second.data(), 
                                                dataMap.at(val.first).at(x).second.size());
                    Z[val.first].first.push_back(arr1);
                    Z[val.first].second.push_back(arr2);
                }
                /* cout << "\n"; */
            }
            /////////////////// debugging 
            /* cout << "Z loaded. contents:\n"; */
            /* for (const auto& z : Z) */
            /* { */
            /*     cout << "zName: " << z.first << "\n"; */
            /*     if (z.second.first.size() != z.second.second.size()) */
            /*     { */
            /*         cout << "values and time not the same size\n"; */
            /*         cout << "values: " << z.second.first.size() << "\n"; */
            /*         cout << "time: " << z.second.second.size() << "\n"; */
            /*         exit(0); */
            /*     } */
            /*     for (unsigned int j = 0; j < z.second.first.size(); ++j) */
            /*     { */
            /*         cout << "sample " << j << " = " ; */
            /*         for (unsigned int k = 0; k < z.second.first.at(j).size(); ++k) */
            /*             cout << z.second.second.at(j)(k) << ":" << z.second.first.at(j)(k) << ","; */
            /*         cout << "\n"; */
            /*     } */
            /*     cout << "---\n"; */
            /* } */
            /* cout << "exiting load_partial_longitudinal\n"; */
        }
    }
}

