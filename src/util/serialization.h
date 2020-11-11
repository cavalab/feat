/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <Eigen/Dense>
#include <Eigen/Core>

#include "../init.h"
#include "json.hpp"
using nlohmann::json;

namespace Eigen
{
    
    /*
     * Serialization for Eigen dynamic types
     */
    template <typename T>
    void to_json(json& j, const MatrixBase<T>& matrix)
    {
        for (int row = 0; row < matrix.rows(); ++row)
        {
            json column = json::array();
            for (int col = 0; col < matrix.cols(); ++col)
            {
                column.push_back(matrix(row, col));
            }
            j.push_back(column);
        }
    }
    
    template <typename T>
    void from_json(const json& j, Matrix<T, Dynamic, Dynamic>& matrix)
    {
        if (j.size() == 0) return;
        matrix.resize(j.size(), j.at(0).size());
        for (std::size_t row = 0; row < j.size(); ++row)
        {
            const auto& jrow = j.at(row);
            for (std::size_t col = 0; col < jrow.size(); ++col)
            {
                const auto& value = jrow.at(col);
                value.get_to(matrix(row, col));
            }

        }
    }

    template <typename T>
    void from_json(const json& j, Matrix<T, Dynamic, 1>& V)
    {
        V.resize(j.size());
        for (int i = 0 ; i < j.size(); ++i)
        {
            j.at(i).get_to(V(i));
        }

    }
}


#endif
