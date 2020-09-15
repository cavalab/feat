/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "../../json.h"
#include "../../init.h"
#include "../pop/op/node.h"

namespace FT{
namespace Pop{
namespace Op{
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Node, name, otype, arity, complexity, visits)
}
}
}
