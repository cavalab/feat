# disutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from eigency.core cimport *

cdef extern from "fewtwo.h" namespace "FT":
    cdef cppclass Fewtwo: 
        Fewtwo(int pop_size, int gens, string ml, 
               bool classification, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               double split, vector[char] dtypes)
        void fit(Map[MatrixXd](X),Map[VectorXd](y))
        VectorXd predict(Map[MatrixXd](X))
        MatrixXd transform(Map[MatrixXd](X))
        VectorXd fit_predict(Map[MatrixXd](X),Map[VectorXd](y))
        VectorXd fit_transform(Map[MatrixXd](X),Map[VectorXd](y))


