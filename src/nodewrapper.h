/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_WRAPPER
#define NODE_WRAPPER

#include <memory>

#ifdef USE_CUDA
    #include "node-cuda/kernels.h"
#endif

#include "node/node.h"

//arithmatic nodes
#include "node/arithmatic/n_2dgaussian.h"
#include "node/arithmatic/n_add.h"
#include "node/arithmatic/n_cos.h"
#include "node/arithmatic/n_cube.h"
#include "node/arithmatic/n_divide.h"
#include "node/arithmatic/n_exponent.h"
#include "node/arithmatic/n_exponential.h"
#include "node/arithmatic/n_float.h"
#include "node/arithmatic/n_gaussian.h"
#include "node/arithmatic/n_log.h"
#include "node/arithmatic/n_logit.h"
#include "node/arithmatic/n_multiply.h"
#include "node/arithmatic/n_sign.h"
#include "node/arithmatic/n_sin.h"
#include "node/arithmatic/n_sqrt.h"
#include "node/arithmatic/n_square.h"
#include "node/arithmatic/n_step.h"
#include "node/arithmatic/n_subtract.h"
#include "node/arithmatic/n_tanh.h"

//control nodes
#include "node/control/n_if.h"
#include "node/control/n_ifthenelse.h"

//learn
#include "node/learn/n_relu.h"
//#include "node/learn/n_split.h"

//logic nodes
#include "node/logic/n_and.h"
#include "node/logic/n_equal.h"
#include "node/logic/n_geq.h"
#include "node/logic/n_greaterthan.h"
#include "node/logic/n_leq.h"
#include "node/logic/n_lessthan.h"
#include "node/logic/n_not.h"
#include "node/logic/n_or.h"
#include "node/logic/n_xor.h"

//longitudinal nodes
#include "node/longitudinal/n_count.h"
#include "node/longitudinal/n_kurtosis.h"
#include "node/longitudinal/n_longitudinal.h"
#include "node/longitudinal/n_max.h"
#include "node/longitudinal/n_mean.h"
#include "node/longitudinal/n_median.h"
#include "node/longitudinal/n_min.h"
#include "node/longitudinal/n_mode.h"
#include "node/longitudinal/n_skew.h"
#include "node/longitudinal/n_slope.h"
#include "node/longitudinal/n_time.h"
#include "node/longitudinal/n_var.h"

//terminal nodes
#include "node/terminals/n_constant.h"
#include "node/terminals/n_variable.h"

//extra include 
//remove after n_split implemented for cuda
#include "./node/n_train.h"

#endif
