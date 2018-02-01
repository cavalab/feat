# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from eigency.core cimport *
cimport numpy as np

cdef extern from "fewtwo.h" namespace "FT":
    cdef cppclass Fewtwo: 
        Fewtwo() except +
        Fewtwo(int pop_size, int gens, string ml, 
               bool classification, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               double split, double fb) except + 
        void fit(Map[MatrixXd] & X, Map[VectorXd] & y)
        VectorXd predict(Map[MatrixXd] & X)
        MatrixXd transform(Map[MatrixXd] & X)
        VectorXd fit_predict(Map[MatrixXd](X),Map[VectorXd](y))
        #VectorXd fit_transform(Map[MatrixXd](X),Map[VectorXd](y))

cdef class PyFewtwo:
    cdef Fewtwo ft  # hold a c++ instance which we're wrapping
    def _cinit_(self,int pop_size, int gens, string ml, 
               bool classification, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, double split, double fb ):
        self.ft = Fewtwo(pop_size,gens,ml,classification,verbosity,max_stall,sel,surv,cross_rate,
                otype, functions, max_depth, max_dim, random_state, erc, obj, shuffle, split, fb)

    def fit(self,np.ndarray X,np.ndarray y):
        self.ft.fit(Map[MatrixXd](X),Map[VectorXd](y))

    def predict(self,np.ndarray X):
        self.ft.predict(Map[MatrixXd](X))

    def transform(self,np.ndarray X):
        return ndarray(self.ft.transform(Map[MatrixXd](X)))

    def fit_predict(self,np.ndarray X,np.ndarray y):
        self.ft.fit(Map[MatrixXd](X), Map[VectorXd](y))
        return ndarray(self.ft.predict(Map[MatrixXd](X)))

