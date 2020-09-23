/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_WRAPPER
#define NODE_WRAPPER

#include <memory>

#ifdef USE_CUDA
    #include "cuda-op/kernels.h"
#endif

#include "op/node.h"
#include "op/n_train.h"
#include "op/n_Dx.h"

//arithmatic nodes
#include "op/arithmetic/n_2dgaussian.h"
#include "op/arithmetic/n_add.h"
#include "op/arithmetic/n_cos.h"
#include "op/arithmetic/n_cube.h"
#include "op/arithmetic/n_divide.h"
#include "op/arithmetic/n_exponent.h"
#include "op/arithmetic/n_exponential.h"
#include "op/arithmetic/n_float.h"
#include "op/arithmetic/n_gaussian.h"
#include "op/arithmetic/n_log.h"
#include "op/arithmetic/n_logit.h"
#include "op/arithmetic/n_multiply.h"
#include "op/arithmetic/n_relu.h"
#include "op/arithmetic/n_sign.h"
#include "op/arithmetic/n_sin.h"
#include "op/arithmetic/n_sqrt.h"
#include "op/arithmetic/n_square.h"
#include "op/arithmetic/n_step.h"
#include "op/arithmetic/n_subtract.h"
#include "op/arithmetic/n_tanh.h"

//control nodes
#include "op/control/n_if.h"
#include "op/control/n_ifthenelse.h"

//learn
#include "op/learn/n_split.h"
#include "op/learn/n_fuzzy_split.h"
#include "op/learn/n_fuzzy_fixed_split.h"

//logic nodes
#include "op/logic/n_and.h"
#include "op/logic/n_equal.h"
#include "op/logic/n_geq.h"
#include "op/logic/n_greaterthan.h"
#include "op/logic/n_leq.h"
#include "op/logic/n_lessthan.h"
#include "op/logic/n_not.h"
#include "op/logic/n_or.h"
#include "op/logic/n_xor.h"

//longitudinal nodes
#include "op/longitudinal/n_count.h"
#include "op/longitudinal/n_kurtosis.h"
#include "op/longitudinal/n_longitudinal.h"
#include "op/longitudinal/n_max.h"
#include "op/longitudinal/n_mean.h"
#include "op/longitudinal/n_median.h"
#include "op/longitudinal/n_min.h"
#include "op/longitudinal/n_recent.h"
#include "op/longitudinal/n_skew.h"
#include "op/longitudinal/n_slope.h"
#include "op/longitudinal/n_var.h"

//terminal nodes
#include "op/terminals/n_constant.h"
#include "op/terminals/n_variable.h"

#endif
