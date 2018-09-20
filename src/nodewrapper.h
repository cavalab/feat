/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_WRAPPER
#define NODE_WRAPPER

#include <memory>

#include "node/node.h"

//boolean nodes
#include "node/bool/n_and.h"
#include "node/bool/n_equal.h"
#include "node/bool/n_geq.h"
#include "node/bool/n_greaterthan.h"
#include "node/bool/n_leq.h"
#include "node/bool/n_lessthan.h"
#include "node/bool/n_not.h"
#include "node/bool/n_or.h"
#include "node/bool/n_split.h"
#include "node/bool/n_xor.h"

//control nodes
#include "node/control/n_if.h"
#include "node/control/n_ifthenelse.h"

//floating nodes
#include "node/float/n_2dgaussian.h"
#include "node/float/n_add.h"
#include "node/float/n_cos.h"
#include "node/float/n_cube.h"
#include "node/float/n_divide.h"
#include "node/float/n_exponent.h"
#include "node/float/n_exponential.h"
#include "node/float/n_float.h"
#include "node/float/n_gaussian.h"
#include "node/float/n_log.h"
#include "node/float/n_logit.h"
#include "node/float/n_multiply.h"
#include "node/float/n_relu.h"
#include "node/float/n_sign.h"
#include "node/float/n_sin.h"
#include "node/float/n_sqrt.h"
#include "node/float/n_square.h"
#include "node/float/n_step.h"
#include "node/float/n_subtract.h"
#include "node/float/n_tanh.h"

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


#endif
