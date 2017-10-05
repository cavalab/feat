


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

} 
