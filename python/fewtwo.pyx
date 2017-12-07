# disutils: language = c++

from libcpp.vector cimport vector

cdef extern from "Fewtwo.h" namespace "FT":
    cdef cppclass Fewtwo: 
        Fewtwo(int pop_size, int gens, string ml, 
               bool classification = false, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               double split, vector[char] dtypes)
        fit(X,y)
        predict(X)
        transform(X)
        fit_predict(X,y)
        fit_transform(X,y)


