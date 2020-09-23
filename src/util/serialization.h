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
/**
 * Provide to_json() and from_json() overloads for nlohmann::json,
 * which allows simple syntax like:
 * 
 * @code
 * Eigen::Matrix3f in, out;
 * 
 * json j;
 * j = in;
 * out = j;
 * @endcode
 * 
 */

namespace Eigen
{
    
    // MatrixBase

    /* template <typename Derived> */
    /* void to_json(json& j, const MatrixBase<Derived>& matrix); */
    
    /* template <typename Derived> */
    /* void from_json(const json& j, MatrixBase<Derived>& matrix); */
    
    // IMPLEMENTATION

    template <typename Derived>
    void to_json(json& j, const MatrixBase<Derived>& matrix)
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
    
    template <typename Derived>
    void from_json(const json& j, MatrixBase<Derived>& matrix)
    {
        using Scalar = typename MatrixBase<Derived>::Scalar;
        
        for (std::size_t row = 0; row < j.size(); ++row)
        {
            const auto& jrow = j.at(row);
            for (std::size_t col = 0; col < jrow.size(); ++col)
            {
                const auto& value = jrow.at(col);
                matrix(row, col) = value.get<Scalar>();
            }
        }
    }
}


#endif
