#ifndef CUDA_TESTS_H
#define CUDA_TESTS_H

#include "testsHeader.h"

#ifdef USE_CUDA
std::map<char, size_t> get_max_state_size(NodeVector &nodes);
bool isValidProgram(NodeVector& program, unsigned num_features);
#endif

#endif
