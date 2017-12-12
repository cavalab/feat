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

cdef class PyFewtwo:
    cdef Fewtwo ft  # hold a c++ instance which we're wrapping
    def _cinit_(self,int pop_size, int gens, string ml, 
               bool classification, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               double split, vector[char] dtypes):
        self.ft = Fewtwo(pop_size,gens,ml,classification,verbosity,max_stall,sel,surv,cross_rate,
                otype, functions, max_depth, max_dim, random_state, erc, obj, shuffle)

    def fit(self,X,y):
        self.ft.fit(X,y)
    def predict(self,X):
        self.ft.predict(X)
    def transform(self,X):
        return ndarray(self.ft.transform(X))
    def fit_predict(self,X,y):
        self.ft.fit(X,y)
        return ndarray(self.ft.predict(y))

