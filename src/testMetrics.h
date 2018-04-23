/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef METRICS_H
#define METRIC_H

#include <Eigen/Dense>

using Eigen::VectorXd;

namespace FT{
namespace metrics{

    /* Scoring functions */

    // Squared difference
    // Derivative of squared difference with respec to yhat
    VectorXd d_squared_difference(const VectorXd& y, const VectorXd& yhat) {
        return 2 * (yhat - y);
    }
} // metrics
} // FT


#endif

