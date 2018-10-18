/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_WRAPPER
#define NODE_WRAPPER

#include <memory>

#include "node/node.h"

//arithmatic nodes
#include "node/arithmetic/n_2dgaussian.h"
#include "node/arithmetic/n_add.h"
#include "node/arithmetic/n_cos.h"
#include "node/arithmetic/n_cube.h"
#include "node/arithmetic/n_divide.h"
#include "node/arithmetic/n_exponent.h"
#include "node/arithmetic/n_exponential.h"
#include "node/arithmetic/n_float.h"
#include "node/arithmetic/n_gaussian.h"
#include "node/arithmetic/n_log.h"
#include "node/arithmetic/n_logit.h"
#include "node/arithmetic/n_multiply.h"
#include "node/arithmetic/n_relu.h"
#include "node/arithmetic/n_sign.h"
#include "node/arithmetic/n_sin.h"
#include "node/arithmetic/n_sqrt.h"
#include "node/arithmetic/n_square.h"
#include "node/arithmetic/n_step.h"
#include "node/arithmetic/n_subtract.h"
#include "node/arithmetic/n_tanh.h"

//control nodes
#include "node/control/n_if.h"
#include "node/control/n_ifthenelse.h"

//learn
#include "node/learn/n_split.h"

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
#include "node/longitudinal/n_recent.h"
#include "node/longitudinal/n_skew.h"
#include "node/longitudinal/n_slope.h"
#include "node/longitudinal/n_time.h"
#include "node/longitudinal/n_var.h"

//terminal nodes
#include "node/terminals/n_constant.h"
#include "node/terminals/n_variable.h"

#endif
