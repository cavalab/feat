/* FEWTWO
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
#include "node/n_add.h"
#include "node/n_and.h"
#include "node/n_geq.h"
#include "node/n_constant.h"
#include "node/n_cos.h"
#include "node/n_cube.h"
#include "node/n_divide.h"
#include "node/n_equal.h"
#include "node/n_exponent.h"
#include "node/n_exp.h"
#include "node/n_greaterthan.h"
#include "node/n_if.h"
#include "node/n_lessthan.h"
#include "node/n_log.h"
#include "node/n_multiply.h"
#include "node/n_not.h"
#include "node/n_leq.h"
#include "node/n_or.h"
#include "node/n_sqrt.h"
#include "node/n_sin.h"
#include "node/n_square.h"
#include "node/n_subtract.h"
#include "node/n_ifthenelse.h"
#include "node/n_variable.h"
#include "node/n_xor.h"
#include "node/n_gaussian.h"
#include "node/n_gaussian2d.h"
#include "node/n_logit.h"
#include "node/n_step.h"
#include "node/n_sign.h"
#include "node/n_tanh.h"
#include "./node/nodelongitudinal.h"
#include "./node/nodemean.h"
#include "./node/nodemedian.h"
#include "./node/nodemax.h"
#include "./node/nodemin.h"
#include "./node/nodevar.h"
#include "./node/nodeskew.h"
#include "./node/nodekurtosis.h"
#include "./node/nodetime.h"
#include "./node/nodeslope.h"
#include "./node/nodemode.h"

#endif
