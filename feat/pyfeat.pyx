# distutils: language=c++

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
import json

cdef extern from "feat.h" namespace "FT":
    cdef cppclass Feat: 
        Feat() except +
        Feat(int pop_size, int gens, string ml, 
               bool classification, int cppclass , int max_stall,
               string sel, string surv, float cross_rate, float root_xo_rate,
               char otype, string functions, 
               unsigned int max_depth, unsigned int max_dim, int random_state, 
               bool erc, string obj,bool shuffle, 
               float split, float fb, string scorer, 
               string feature_names, bool backprop, int iters, float lr, 
               int batch_size, int n_jobs, bool hillclimb, string logfile, 
               int max_time, bool residual_xo, bool stagewise_xo, 
               bool stagewise_xo_tol, bool softmax_norm, int save_pop, 
               bool normalize, bool val_from_arch, bool corr_delete_mutate, 
               float simplify, string protected_groups, bool tune_initial,
               bool tune_final, string starting_pop 
               ) except + 

        void fit(float * X, int rowsX, int colsX, float*  y , int lenY)
        VectorXd predict(float * X, int rowsX, int colsX)
        VectorXd predict_archive(int i, float * X, int rowsX, int colsX)
        ArrayXXd predict_proba_archive(int i, float * X, int rowsX, int colsX)
        ArrayXXd predict_proba(float * X, int rowsX, int colsX)
        MatrixXd transform(float * X, int rowsX, int colsX)
        VectorXd fit_predict(float * X, int rowsX, int colsX, float * y, 
                int lenY)
        MatrixXd fit_transform(float * X, int rowsX, int colsX, float * y, 
                int lenY)
        string get_representation()
        string get_archive(bool front)
        string get_model(bool)
        string get_eqn(bool)

        void fit_with_z(float * X,int rowsX,int colsX, float * Y,int lenY, 
                string s, int * train_idx, int train_size)
        VectorXd predict_with_z(float * X,int rowsX,int colsX, string s, 
                            int * idx, int idx_size)
        MatrixXd transform_with_z(float * X,int rowsX,int colsX, string s, 
                            int * train_idx, int train_size)
        ArrayXXd predict_proba_with_z(float * X,int rowsX,int colsX, string s, 
                            int * idx, int idx_size)

        ArrayXd get_coefs()
        int get_n_params()
        int get_complexity()
        int get_dim()
        int get_n_nodes()
        int get_archive_size()
        int get_pop_size()
        void set_pop_size(int)
        int get_gens()
        void set_gens(int)
        string get_ml()
        void set_ml(string)
        bool get_classification()
        void set_classification(bool)
        int get_verbosity()
        void set_verbosity(int)
        int get_max_stall()
        void set_max_stall(int)
        string get_sel()
        void set_sel(string)
        string get_surv()
        void set_surv(string )
        float get_cross_rate()
        void set_cross_rate(float)
        float get_root_xo_rate()
        void set_root_xo_rate(float)
        char get_otype()
        void set_otype(char )
        string get_functions()
        string get_functions_()
        void set_functions(string )
        int get_max_depth()   
        void set_max_depth(int)
        int get_max_dim()  
        void set_max_dim(int )
        int get_random_state()
        void set_random_state(int)
        bool get_erc()  
        void set_erc(bool)
        string get_obj()
        void set_obj(string)
        bool get_shuffle()  
        void set_shuffle(bool)
        float get_split()  
        void set_split(float)
        float get_fb()
        void set_fb(float)
        string get_scorer()
        void set_scorer(string)
        string get_feature_names()
        void set_feature_names(string)
        bool get_backprop()
        void set_backprop(bool)
        int get_iters()
        void set_iters(int)
        float get_lr()
        void set_lr(float)
        int get_batch_size()
        void set_batch_size(int)
        int get_n_jobs()
        void set_n_jobs(int)
        bool get_hillclimb()
        void set_hillclimb(bool)
        string get_logfile()        
        void set_logfile(string)
        int get_max_time()
        void set_max_time(int)
        bool get_residual_xo()
        void set_residual_xo(bool)
        bool get_stagewise_xo()
        void set_stagewise_xo(bool)
        float get_stagewise_xo_tol()
        void set_stagewise_xo_tol(float)
        bool get_softmax_norm()
        void set_softmax_norm(bool)
        bool get_save_pop()
        void set_save_pop(bool)
        bool get_normalize()
        void set_normalize(bool)
        bool get_val_from_arch()
        void set_val_from_arch(bool)
        bool get_corr_delete_mutate()
        void set_corr_delete_mutate(bool)
        float get_simplify()
        void set_simplify(float)
        string get_protected_groups()
        void set_protected_groups(string)
        bool get_tune_initial()
        void set_tune_initial(bool)
        bool get_tune_final() 
        void set_tune_final(bool)
        string get_starting_pop()
        void set_starting_pop(string)

        string get_stats()
        
        void load(string filename)
        string save()
        void load_best_ind(string filename)
        void load_population(string filename)
        bool fitted

cdef class PyFeat:
    
    cdef Feat ft  # hold a c++ instance which we're wrapping
    def __cinit__(self, **kwargs):

        self.ft = Feat()
        # self._set_params(**kwargs)

    def _fit(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)

        self.ft.fit(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y))

    def _fit_with_z(self,np.ndarray X,np.ndarray y, str zfile, 
            np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)

        self.ft.fit_with_z(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],
                len(arr_y), zfile.encode(), &arr_z_id[0], len(arr_z_id))
                           
    def _transform_with_z(self,np.ndarray X, str zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        res = ndarray(self.ft.transform_with_z(&arr_x[0,0],X.shape[0],
            X.shape[1], zfile.encode(), &arr_z_id[0], len(arr_z_id)))
        return res.transpose()


    def _predict(self,np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        res = ndarray(self.ft.predict(&arr_x[0,0],X.shape[0],X.shape[1]))
        return res.flatten()

    def _predict_archive(self,int i, np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        res = ndarray(self.ft.predict_archive(i, &arr_x[0,0],
            X.shape[0],X.shape[1]))
        return res

    def _predict_proba_archive(self,int i, np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        res = ndarray(self.ft.predict_proba_archive(i, &arr_x[0,0],
            X.shape[0],X.shape[1]))
        return np.transpose(res)

    def _predict_with_z(self,np.ndarray X, str zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_with_z(&arr_x[0,0],X.shape[0],X.shape[1],
                                 zfile.encode(), &arr_z_id[0], len(arr_z_id)))
        return res.flatten()

    def _predict_proba(self,np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        res = ndarray(self.ft.predict_proba(&arr_x[0,0],X.shape[0],X.shape[1]))
        return np.transpose(res)

    def _predict_proba_with_z(self,np.ndarray X, str zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_proba_with_z(&arr_x[0,0],
            X.shape[0],X.shape[1], zfile.encode(), &arr_z_id[0], len(arr_z_id)))
        return np.transpose(res)

    def _transform(self,np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        X = ndarray(self.ft.transform(&arr_x[0,0],X.shape[0],X.shape[1]))
        return X.transpose()

    def _fit_predict(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        
        return ndarray(self.ft.fit_predict( &arr_x[0,0],X.shape[0],
            X.shape[1],&arr_y[0],len(arr_y) ))

    def _fit_transform(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        
        X = ndarray(self.ft.fit_transform( &arr_x[0,0],X.shape[0],X.shape[1],
            &arr_y[0],len(arr_y) ))
        return X.transpose()
    
    def get_representation(self):
        return self.ft.get_representation().decode()

    def get_archive(self,justfront=False):
        """Returns all the final representation equations in the archive"""
        archive = []
        str_arc = self.ft.get_archive(justfront).decode()
        for model in str_arc.splitlines():
            archive.append( json.loads(model))
        return archive

    def get_model(self, bool sort=True):
        return self.ft.get_model(sort).decode()

    def get_eqn(self, bool sort=True):
        """best model as a single line equation"""
        return self.ft.get_eqn(sort).decode()

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
        
    def get_archive_size(self):
        return self.ft.get_archive_size()

    def _load(self, feat_state):
        self.ft.load(feat_state.encode())

    def _save(self):
        return self.ft.save().decode()

    def _load_best_ind(self, filename):
        self.ft.load_best_ind(filename.encode())

    def _load_population(self, filename):
        self.ft.load_population(filename)

    def _set_params(self, **params):
        # print('_set_params called with',params)
        for k,v in params.items():
            if not k.endswith('_'):
                setattr(self, k, v)

    def _get_params(self):
        exclusions = ['_repr_html_']
        property_names=[]
        for p in dir(self.__class__):
            if ((not p.startswith('_')) 
                or p.startswith('__') 
                ): 
                continue
            if p in exclusions:
                continue
            # note: cython properties are actually basic data types
            if type(getattr(self,p)).__name__ in ['bool','int','str','float']:
                property_names.append(p)
            # else:
            #     print(p,'rejected, type name:',type(getattr(self,p)).__name__)
        tmp = {p:getattr(self,p) for p in property_names}
        # print('returning',tmp,'from _get_params()')
        return tmp
        # return json.loads(self.ft.get_params())

    ###########################################################################
    # decorated property setters and getters
    ###########################################################################
    @property
    def stats_(self):
        return json.loads(self.ft.get_stats())

    @property
    def _pop_size(self):  
        return self.ft.get_pop_size()
    @_pop_size.setter
    def _pop_size(self, value):
        self.ft.set_pop_size(value)

    @property
    def _gens(self):  
        return self.ft.get_gens()
    @_gens.setter
    def _gens(self, value):
        self.ft.set_gens(value)

    @property
    def _ml(self): 
        return self.ft.get_ml().decode()
    @_ml.setter
    def _ml(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_ml(value)

    @property
    def _classification(self):  
        return self.ft.get_classification()
    @_classification.setter
    def _classification(self, value):
        self.ft.set_classification(value)

    @property
    def _verbosity(self):  
        return self.ft.get_verbosity()
    @_verbosity.setter
    def _verbosity(self, value):
        self.ft.set_verbosity(value)

    @property
    def _max_stall(self):
        return self.ft.get_max_stall()
    @_max_stall.setter
    def _max_stall(self, value):
        self.ft.set_max_stall(value)

    @property
    def _sel(self):  
        return self.ft.get_sel().decode()
    @_sel.setter
    def _sel(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_sel(value)

    @property
    def _surv(self):  
        return self.ft.get_surv().decode()
    @_surv.setter
    def _surv(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_surv(value)

    @property
    def _cross_rate(self): 
        return self.ft.get_cross_rate()
    @_cross_rate.setter
    def _cross_rate(self, value):
        self.ft.set_cross_rate(value)

    @property
    def _root_xo_rate(self):
        return self.ft.get_root_xo_rate()
    @_root_xo_rate.setter
    def _root_xo_rate(self, value):
        self.ft.set_root_xo_rate(value)

    @property
    def _otype(self):  
        return string(1, <char>self.ft.get_otype()).decode()
        # return str(self.ft.get_otype()).encode()
    @_otype.setter
    def _otype(self, value):
        self.ft.set_otype(ord(value))

    @property
    def _functions_(self): 
        return self.ft.get_functions_().decode()
    @property
    def _functions(self): 
        return self.ft.get_functions().decode()
    @_functions.setter
    def _functions(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_functions(value)

    @property
    def _max_depth(self):   
        return self.ft.get_max_depth()
    @_max_depth.setter
    def _max_depth(self, value):
        self.ft.set_max_depth(value)

    @property
    def _max_dim(self):  
        return self.ft.get_max_dim()
    @_max_dim.setter
    def _max_dim(self, value):
        self.ft.set_max_dim(value)

    @property
    def _random_state(self): 
        return self.ft.get_random_state()
    @_random_state.setter
    def _random_state(self, value):
        self.ft.set_random_state(value)

    @property
    def _erc(self):  
        return self.ft.get_erc()
    @_erc.setter
    def _erc(self, value):
        self.ft.set_erc(value)

    @property
    def _obj(self): 
        return self.ft.get_obj().decode()
    @_obj.setter
    def _obj(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_obj(value)

    @property
    def _shuffle(self):  
        return self.ft.get_shuffle()
    @_shuffle.setter
    def _shuffle(self, value):
        self.ft.set_shuffle(value)

    @property
    def _split(self):  
        return self.ft.get_split()
    @_split.setter
    def _split(self, value):
        self.ft.set_split(value)

    @property
    def _fb(self):
        return self.ft.get_fb()
    @_fb.setter
    def _fb(self, value):
        self.ft.set_fb(value)

    @property
    def _scorer(self):
        return self.ft.get_scorer().decode()
    @_scorer.setter
    def _scorer(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_scorer(value)

    @property
    def _feature_names(self):
        return self.ft.get_feature_names().decode()
    @_feature_names.setter
    def _feature_names(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_feature_names(value)

    @property
    def _backprop(self):
        return self.ft.get_backprop()
    @_backprop.setter
    def _backprop(self, value):
        self.ft.set_backprop(value)

    @property
    def _iters(self):
        return self.ft.get_iters()
    @_iters.setter
    def _iters(self, value):
        self.ft.set_iters(value)

    @property
    def _lr(self):
        return self.ft.get_lr()
    @_lr.setter
    def _lr(self, value):
        self.ft.set_lr(value)

    @property
    def _batch_size(self):
        return self.ft.get_batch_size()
    @_batch_size.setter
    def _batch_size(self, value):
        self.ft.set_batch_size(value)

    @property
    def _n_jobs(self):
        return self.ft.get_n_jobs()
    @_n_jobs.setter
    def _n_jobs(self, value):
        self.ft.set_n_jobs(value)

    @property
    def _hillclimb(self):
        return self.ft.get_hillclimb()
    @_hillclimb.setter
    def _hillclimb(self, value):
        self.ft.set_hillclimb(value)

    @property
    def _logfile(self):
        return self.ft.get_logfile().decode()
    @_logfile.setter
    def _logfile(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_logfile(value)

    @property
    def _max_time(self):
        return self.ft.get_max_time()
    @_max_time.setter
    def _max_time(self, value):
        self.ft.set_max_time(value)

    @property
    def _residual_xo(self):
        return self.ft.get_residual_xo()
    @_residual_xo.setter
    def _residual_xo(self, value):
        self.ft.set_residual_xo(value)

    @property
    def _stagewise_xo(self):
        return self.ft.get_stagewise_xo()
    @_stagewise_xo.setter
    def _stagewise_xo(self, value):
        self.ft.set_stagewise_xo(value)

    @property
    def _stagewise_xo_tol(self):
        return self.ft.get_stagewise_xo_tol()
    @_stagewise_xo_tol.setter
    def _stagewise_xo_tol(self, value):
        self.ft.set_stagewise_xo_tol(value)

    @property
    def _softmax_norm(self):
        return self.ft.get_softmax_norm()
    @_softmax_norm.setter
    def _softmax_norm(self, value):
        self.ft.set_softmax_norm(value)

    @property
    def _save_pop(self):
        return self.ft.get_save_pop()
    @_save_pop.setter
    def _save_pop(self, value):
        self.ft.set_save_pop(value)

    @property
    def _normalize(self):
        return self.ft.get_normalize()
    @_normalize.setter
    def _normalize(self, value):
        self.ft.set_normalize(value)

    @property
    def _val_from_arch(self):
        return self.ft.get_val_from_arch()
    @_val_from_arch.setter
    def _val_from_arch(self, value):
        self.ft.set_val_from_arch(value)

    @property
    def _corr_delete_mutate(self):
        return self.ft.get_corr_delete_mutate()
    @_corr_delete_mutate.setter
    def _corr_delete_mutate(self, value):
        self.ft.set_corr_delete_mutate(value)

    @property
    def _simplify(self):
        return self.ft.get_simplify()
    @_simplify.setter
    def _simplify(self, value):
        self.ft.set_simplify(value)

    @property
    def _protected_groups(self):
        return self.ft.get_protected_groups().decode()
    @_protected_groups.setter
    def _protected_groups(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_protected_groups(value)

    @property
    def _tune_initial(self):
        return self.ft.get_tune_initial()
    @_tune_initial.setter
    def _tune_initial(self, value):
        self.ft.set_tune_initial(value)

    @property
    def _tune_final(self): 
        return self.ft.get_tune_final()
    @_tune_final.setter
    def _tune_final(self, value):
        self.ft.set_tune_final(value)

    @property
    def _starting_pop(self):
        return self.ft.get_starting_pop().decode()
    @_starting_pop.setter
    def _starting_pop(self, value):
        if isinstance(value,str): value = value.encode() 
        self.ft.set_starting_pop(value)

    @property
    def _fitted_(self):
        return self.ft.fitted
