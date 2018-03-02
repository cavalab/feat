# distutils: language=c++
cimport numpy 
import numpy 
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from eigency.core cimport *
from sklearn.utils import check_X_y

cdef extern from "feat.h" namespace "FT":
    cdef cppclass Feat: 
        Feat() except +
        Feat(int pop_size, int gens, string ml, 
               bool classification, int verbosity, int max_stall,
               string sel, string surv, float cross_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               double split, double fb) except + 
        void fit(double * X, int rowsX, int colsX, double*  y , int lenY)
        VectorXd predict(double * X, int rowsX,int colsX)
        MatrixXd transform(double * X, int rowsX,int colsX)
        VectorXd fit_predict(double * X, int rowsX,int colsX, double*  y , int lenY)
        MatrixXd fit_transform(double * X, int rowsX,int colsX, double*  y , int lenY)

cdef class PyFeat:
    cdef Feat ft  # hold a c++ instance which we're wrapping
    def __cinit__(self,int pop_size, int gens, string ml, bool classification, int verbosity, int max_stall,string sel, string surv, float cross_rate,string otype, string functions, unsigned int max_depth, unsigned int max_dim, int random_state, bool erc , string obj,bool shuffle, double split, double fb):
        cdef char otype_char
        if ( len(otype) == 0):
            otype_char = 'a' #Defaut Value
        else:
            otype_char = ord(otype)
        self.ft = Feat(pop_size,gens,ml,classification,verbosity,max_stall,sel,surv,cross_rate,
otype_char, functions, max_depth, max_dim, random_state, erc, obj, shuffle, split, fb)
    def fit(self,numpy.ndarray X,numpy.ndarray y):

        cdef numpy.ndarray[numpy.double_t, ndim=2, mode="c"] arr_x
        cdef numpy.ndarray[numpy.double_t, ndim=1, mode="c"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = numpy.ascontiguousarray(X, dtype=numpy.double)
        arr_y = numpy.ascontiguousarray(y, dtype=numpy.double)
 
        cdef int c_rows = X.shape[0]
        cdef int c_cols = X.shape[1]
        cdef int c_rows_y = len(arr_y)
        self.ft.fit(&arr_x[0,0],c_rows,c_cols,&arr_y[0],c_rows_y)

    def predict(self,numpy.ndarray X):
        cdef numpy.ndarray[numpy.double_t, ndim=2, mode="c"] arr_x
        X = X.transpose()
        arr_x = numpy.ascontiguousarray(X, dtype=numpy.double)

        cdef int c_rows = X.shape[0]
        cdef int c_cols = X.shape[1]
        res = ndarray(self.ft.predict(&arr_x[0,0],c_rows,c_cols))
        return res

    def transform(self,np.ndarray X):
        cdef numpy.ndarray[numpy.double_t, ndim=2, mode="c"] arr_x
        X = X.transpose()
        arr_x = numpy.ascontiguousarray(X, dtype=numpy.double)

        cdef int c_rows = X.shape[0]
        cdef int c_cols = X.shape[1]
        X = ndarray(self.ft.transform(&arr_x[0,0],c_rows,c_cols))
        return X.transpose()

    def fit_predict(self,np.ndarray X,np.ndarray y):
        cdef numpy.ndarray[numpy.double_t, ndim=2, mode="c"] arr_x
        cdef numpy.ndarray[numpy.double_t, ndim=1, mode="c"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = numpy.ascontiguousarray(X, dtype=numpy.double)
        arr_y = numpy.ascontiguousarray(y, dtype=numpy.double)
        
        cdef int c_rows = X.shape[0]
        cdef int c_cols = X.shape[1]
        cdef int c_rows_y = len(arr_y)
        return ndarray(self.ft.fit_predict( &arr_x[0,0],c_rows,c_cols,&arr_y[0],c_rows_y ))

    def fit_transform(self,np.ndarray X,np.ndarray y):
        cdef numpy.ndarray[numpy.double_t, ndim=2, mode="c"] arr_x
        cdef numpy.ndarray[numpy.double_t, ndim=1, mode="c"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = numpy.ascontiguousarray(X, dtype=numpy.double)
        arr_y = numpy.ascontiguousarray(y, dtype=numpy.double)
        
        cdef int c_rows = X.shape[0]
        cdef int c_cols = X.shape[1]
        cdef int c_rows_y = len(arr_y)
        X = ndarray(self.ft.fit_transform( &arr_x[0,0],c_rows,c_cols,&arr_y[0],c_rows_y ))
        return X.transpose()

