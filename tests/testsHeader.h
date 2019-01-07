#ifndef TESTS_HEADER_H
#define TESTS_HEADER_H

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <shogun/base/init.h>
#include <omp.h>
#include <string>
#include <stack>
#include <gtest/gtest.h>	

// stuff being used

using namespace std;

using Eigen::MatrixXf;
using Eigen::VectorXf;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::stoi;
using std::to_string;
using std::stof;
using namespace shogun;

#define private public

#include <cstdio>
#include "../src/feat.h"

using namespace FT;

#endif
