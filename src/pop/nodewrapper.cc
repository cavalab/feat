#include "nodewrapper.h"

namespace FT{
namespace Pop{
namespace Op{

map<string, Node*> node_map = {
    { "+",  new NodeAdd({1.0,1.0})}, 
    { "-",  new NodeSubtract({1.0,1.0})}, 
    { "*",  new NodeMultiply({1.0,1.0})}, 
    { "/",  new NodeDivide({1.0,1.0})}, 
    { "sqrt",  new NodeSqrt({1.0})}, 
    { "sin",  new NodeSin({1.0})}, 
    { "cos",  new NodeCos({1.0})}, 
    { "tanh",  new NodeTanh({1.0})}, 
    { "^2",  new NodeSquare({1.0})}, 
    { "^3",  new NodeCube({1.0})}, 
    { "^",  new NodeExponent({1.0})}, 
    { "exp",  new NodeExponential({1.0})}, 
    { "gauss",  new NodeGaussian({1.0})}, 
    { "gauss2d",  new Node2dGaussian({1.0, 1.0})}, 
    { "log", new NodeLog({1.0}) },   
    { "logit", new NodeLogit({1.0}) },
    { "relu", new NodeRelu({1.0}) },
    { "b2f", new NodeFloat<bool>() },
    { "c2f", new NodeFloat<int>() },
    
    // logical operators
    { "and", new NodeAnd() },
    { "or", new NodeOr() },
    { "not", new NodeNot() },
    { "xor", new NodeXor() },
    { "=", new NodeEqual() },
    { ">", new NodeGreaterThan() },
    { ">=", new NodeGEQ() },        
    { "<", new NodeLessThan() },
    { "<=", new NodeLEQ() },
    { "split", new NodeSplit<float>() },
    { "fuzzy_split", new NodeFuzzySplit<float>() },
    { "fuzzy_fixed_split", new NodeFuzzyFixedSplit<float>() },
    { "split_c", new NodeSplit<int>() },
    { "fuzzy_split_c", new NodeFuzzySplit<int>() },
    { "fuzzy_fixed_split_c", new NodeFuzzyFixedSplit<int>() },
    { "if", new NodeIf() },   	    		
    { "ite", new NodeIfThenElse() },
    { "step", new NodeStep() },
    { "sign", new NodeSign() },
       
    // longitudinal nodes
    { "mean", new NodeMean() },
    { "median", new NodeMedian() },
    { "max", new NodeMax() },
    { "min", new NodeMin() },
    { "variance", new NodeVar() },
    { "skew", new NodeSkew() },
    { "kurtosis", new NodeKurtosis() },
    { "slope", new NodeSlope() },
    { "count", new NodeCount() },
    { "recent", new NodeRecent() },

    // terminals
    { "variable_f", new NodeVariable<float>() },
    { "variable_b", new NodeVariable<bool>() },
    { "variable_c", new NodeVariable<int>() },
    { "constant_b", new NodeConstant() },
    { "constant_d", new NodeConstant() },
    /* { "z", new NodeLongitudinal() } */
}; //node_map
} // Op
} // Pop
} // FT
