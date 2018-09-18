# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

# distutils: language=c++
import ctypes
import numpy as np 
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
               double split, double fb, string scorer, 
               string feature_names, bool backprop, int iters, double lr, int bs,
               int n_threads, bool hillclimb, string logfile, int max_time, bool use_batch,
               bool semantic_xo) except + 

        void fit(double * X, int rowsX, int colsX, double*  y , int lenY)
        VectorXd predict(double * X, int rowsX,int colsX)
        ArrayXXd predict_proba(double * X, int rowsX,int colsX)
        MatrixXd transform(double * X, int rowsX,int colsX)
        VectorXd fit_predict(double * X, int rowsX,int colsX, double*  y , int lenY)
        MatrixXd fit_transform(double * X, int rowsX,int colsX, double*  y , int lenY)
        string get_representation()
        string get_eqns()
        string get_model()

        void fit_with_z(double * X,int rowsX,int colsX, double * Y,int lenY, string s, 
                            int * train_idx, int train_size)
        VectorXd predict_with_z(double * X,int rowsX,int colsX, string s, 
                            int * idx, int idx_size)
        MatrixXd transform_with_z(double * X,int rowsX,int colsX, string s, 
                            int * train_idx, int train_size)
        ArrayXXd predict_proba_with_z(double * X,int rowsX,int colsX, string s, 
                            int * idx, int idx_size)
        ArrayXd get_coefs()
        int get_n_params()
        int get_complexity()
        int get_dim()
        int get_n_nodes()

cdef class PyFeat:
    cdef Feat ft  # hold a c++ instance which we're wrapping
    def __cinit__(self,int pop_size, int gens, string ml, bool classification, int verbosity, 
                  int max_stall,string sel, string surv, float cross_rate,string otype, 
                  string functions, unsigned int max_depth, unsigned int max_dim, 
                  int random_state, 
                  bool erc , string obj, bool shuffle, double split, double fb,
                  string scorer, string feature_names, bool backprop, int iters, double lr, int bs,
                  int n_threads, bool hillclimb, string logfile, int max_time, bool use_batch,
                  bool semantic_xo):
        
        cdef char otype_char
        if ( len(otype) == 0):
            otype_char = 'a' #Defaut Value
        else:
            otype_char = ord(otype)
        self.ft = Feat(pop_size,gens,ml,classification,verbosity,max_stall,sel,surv,cross_rate,
        otype_char, functions, max_depth, max_dim, random_state, erc, obj, shuffle, split, fb, scorer,
        feature_names, backprop, iters, lr, bs, n_threads, hillclimb, logfile, max_time, 
        use_batch, semantic_xo)

    def fit(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.double_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_y = np.asfortranarray(y, dtype=np.double)

        self.ft.fit(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y))

    def fit_with_z(self,np.ndarray X,np.ndarray y, string zfile, np.ndarray zids):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.double_t, ndim=1, mode="fortran"] arr_y
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_y = np.asfortranarray(y, dtype=np.double)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)

        self.ft.fit_with_z(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y),
                           zfile, &arr_z_id[0], len(arr_z_id))
                           
    def transform_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        res = ndarray(self.ft.transform_with_z(&arr_x[0,0],X.shape[0],X.shape[1], zfile, &arr_z_id[0], len(arr_z_id)))
        return res.transpose()


    def predict(self,np.ndarray X):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)

        res = ndarray(self.ft.predict(&arr_x[0,0],X.shape[0],X.shape[1]))
        return res.flatten()

    def predict_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_with_z(&arr_x[0,0],X.shape[0],X.shape[1],
                                             zfile, &arr_z_id[0], len(arr_z_id)))
        return res.flatten()

    def predict_proba(self,np.ndarray X):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)

        res = ndarray(self.ft.predict_proba(&arr_x[0,0],X.shape[0],X.shape[1]))
        return res.flatten()

    def predict_proba_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_proba_with_z(&arr_x[0,0],X.shape[0],X.shape[1],
                                             zfile, &arr_z_id[0], len(arr_z_id)))
        return res.flatten()

    def transform(self,np.ndarray X):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)

        X = ndarray(self.ft.transform(&arr_x[0,0],X.shape[0],X.shape[1]))
        return X.transpose()

    def fit_predict(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.double_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_y = np.asfortranarray(y, dtype=np.double)
        
        return ndarray(self.ft.fit_predict( &arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y) ))

    def fit_transform(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.double_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.double_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.double)
        arr_y = np.asfortranarray(y, dtype=np.double)
        
        X = ndarray(self.ft.fit_transform( &arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y) ))
        return X.transpose()
    
    def get_representation(self):
        return self.ft.get_representation().decode()

    def get_archive(self):
        return self.ft.get_eqns().decode()

    def get_model(self):
        return self.ft.get_model().decode()

    def get_coefs(self):
        return ndarray(self.ft.get_coefs()).flatten()
    
    def get_n_params(self):
        return self.ft.get_n_params()

    def get_complexity(self):
        return self.ft.get_complexity()

    def get_dim(self):
        return self.ft.get_dim()

    def get_n_nodes(self):
        return self.ft.get_n_nodes()
