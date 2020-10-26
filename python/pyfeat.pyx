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
from sklearn.utils import check_X_y
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
               int batch_size, int n_threads, bool hillclimb, string logfile, 
               int max_time, bool residual_xo, bool stagewise_xo, 
               bool stagewise_xo_tol, bool softmax_norm, int print_pop, 
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
        string get_model()

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
        int get_n_threads()
        void set_n_threads(int)
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
        
        # vector[int] get_gens();        
        # vector[float] get_timers();
        # vector[float] get_min_losses();
        # vector[float] get_min_losses_val();
        # vector[float] get_med_scores();
        # vector[float] get_med_loss_vals();
        # vector[unsigned] get_med_size();
        # vector[unsigned] get_med_complexities();
        # vector[unsigned] get_med_num_params();
        # vector[unsigned] get_med_dim();
        void load(string filename)
        void save(string filename)
        void load_best_ind(string filename)
        void load_population(string filename)

#@cython.auto_pickle(True)
cdef class PyFeat:
    cdef Feat ft  # hold a c++ instance which we're wrapping
    def __cinit__(self, int pop_size, int gens, string ml, bool classification, 
            int verbosity, int max_stall,string sel, string surv, 
            float cross_rate, float root_xo_rate, string otype, 
            string functions, unsigned int max_depth, unsigned int max_dim, 
            int random_state, bool erc , string obj, bool shuffle, float split, 
            float fb, string scorer, string feature_names, bool backprop, 
            int iters, float lr, int batch_size, int n_threads, bool hillclimb, 
            string logfile, int max_time, bool residual_xo, 
            bool stagewise_xo, bool stagewise_xo_tol, bool softmax_norm, 
            int print_pop, bool normalize, bool val_from_arch, 
            bool corr_delete_mutate, float simplify, string protected_groups,
            bool tune_initial, bool tune_final, string starting_pop):
        
        cdef char otype_char
        if ( len(otype) == 0):
            otype_char = 'a' #Defaut Value
        else:
            otype_char = ord(otype)
        self.ft = Feat(pop_size,gens,ml,classification, verbosity,max_stall,sel,
                surv, cross_rate, root_xo_rate, otype_char, functions, 
                max_depth, max_dim, random_state, erc, obj, shuffle, split, fb,
                scorer, feature_names, backprop, iters, lr, batch_size, 
                n_threads, hillclimb, logfile, max_time, residual_xo, 
                stagewise_xo, stagewise_xo_tol, softmax_norm, print_pop, 
                normalize, val_from_arch, corr_delete_mutate, simplify,
                protected_groups, tune_initial, tune_final, starting_pop)

    def _fit(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)

        self.ft.fit(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],len(arr_y))

    def _fit_with_z(self,np.ndarray X,np.ndarray y, string zfile, 
            np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)

        self.ft.fit_with_z(&arr_x[0,0],X.shape[0],X.shape[1],&arr_y[0],
                len(arr_y), zfile, &arr_z_id[0], len(arr_z_id))
                           
    def _transform_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        res = ndarray(self.ft.transform_with_z(&arr_x[0,0],X.shape[0],
            X.shape[1], zfile, &arr_z_id[0], len(arr_z_id)))
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
        return res

    def _predict_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_with_z(&arr_x[0,0],X.shape[0],X.shape[1],
                                         zfile, &arr_z_id[0], len(arr_z_id)))
        return res.flatten()

    def _predict_proba(self,np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        res = ndarray(self.ft.predict_proba(&arr_x[0,0],X.shape[0],X.shape[1]))
        return res.flatten()

    def _predict_proba_with_z(self,np.ndarray X, string zfile, np.ndarray zids):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[int, ndim=1, mode="fortran"] arr_z_id
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_z_id = np.asfortranarray(zids, dtype=ctypes.c_int)
        
        res = ndarray(self.ft.predict_proba_with_z(&arr_x[0,0],
            X.shape[0],X.shape[1], zfile, &arr_z_id[0], len(arr_z_id)))
        return res.flatten()

    def _transform(self,np.ndarray X):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)

        X = ndarray(self.ft.transform(&arr_x[0,0],X.shape[0],X.shape[1]))
        return X.transpose()

    def _fit_predict(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        
        return ndarray(self.ft.fit_predict( &arr_x[0,0],X.shape[0],
            X.shape[1],&arr_y[0],len(arr_y) ))

    def _fit_transform(self,np.ndarray X,np.ndarray y):
        cdef np.ndarray[np.float32_t, ndim=2, mode="fortran"] arr_x
        cdef np.ndarray[np.float32_t, ndim=1, mode="fortran"] arr_y
        check_X_y(X,y,ensure_2d=True,ensure_min_samples=1)
        X = X.transpose()
        arr_x = np.asfortranarray(X, dtype=np.float32)
        arr_y = np.asfortranarray(y, dtype=np.float32)
        
        X = ndarray(self.ft.fit_transform( &arr_x[0,0],X.shape[0],X.shape[1],
            &arr_y[0],len(arr_y) ))
        return X.transpose()
    
    def representation(self):
        return self.ft.get_representation().decode()

    def _get_archive(self,justfront):
        return self.ft.get_archive(justfront).decode()

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
        
    @property
    def stats(self):
        return json.loads(self.ft.get_stats())
        
    # def get_timers(self):
    #     return self.ft.get_stats_timers()
        
    # def get_min_losses(self):
    #     return self.ft.get_min_losses()
        
    # def get_min_losses_val(self):
    #     return self.ft.get_min_losses_val()
        
    # def get_med_scores(self):
    #     return self.ft.get_med_scores()
        
    # def get_med_loss_vals(self):
    #     return self.ft.get_med_loss_vals()
        
    # def get_med_size(self):
    #     return self.ft.get_med_size()
        
    # def get_med_complexities(self):
    #     return self.ft.get_med_complexities()
        
    # def get_med_num_params(self):
    #     return self.ft.get_med_num_params()
        
    # def get_med_dim(self):
    #     return self.ft.get_med_dim()
   
    def get_archive_size(self):
        return self.ft.get_archive_size()

    def load(self, filename):
        self.ft.load(str(filename).encode())

    def save(self, filename):
        self.ft.save(str(filename).encode())

    def load_best_ind(self, filename):
        self.ft.load_best_ind(str(filename).encode())

    def _load_population(self, filename):
        self.ft.load_population(filename)


    ###########################################################################
    # setters and getters
    ###########################################################################
    @property
    def pop_size(self):  
        return self.ft.get_pop_size()
    @pop_size.setter
    def pop_size(self, value):
        self.ft.set_pop_size(value)

    @property
    def gens(self):  
        return self.ft.get_gens()
    @gens.setter
    def gens(self, value):
        self.ft.set_gens(value)

    @property
    def ml(self): 
        return self.ft.get_ml()
    @ml.setter
    def ml(self, value):
        self.ft.set_ml(value)

    @property
    def classification(self):  
        return self.ft.get_classification()
    @classification.setter
    def classification(self, value):
        self.ft.set_classification(value)

    @property
    def verbosity(self):  
        return self.ft.get_verbosity()
    @verbosity.setter
    def verbosity(self, value):
        self.ft.set_verbosity(value)

    @property
    def max_stall(self):
        return self.ft.get_max_stall()
    @max_stall.setter
    def max_stall(self, value):
        self.ft.set_max_stall(value)

    @property
    def sel(self):  
        return self.ft.get_sel()
    @sel.setter
    def sel(self, value):
        self.ft.set_sel(value)

    @property
    def surv(self):  
        return self.ft.get_surv()
    @surv.setter
    def surv(self, value):
        self.ft.set_surv(value)

    @property
    def cross_rate(self): 
        return self.ft.get_cross_rate()
    @cross_rate.setter
    def cross_rate(self, value):
        self.ft.set_cross_rate(value)

    @property
    def root_xo_rate(self):
        return self.ft.get_root_xo_rate()
    @root_xo_rate.setter
    def root_xo_rate(self, value):
        self.ft.set_root_xo_rate(value)

    @property
    def otype(self):  
        return str(self.ft.get_otype())
    @otype.setter
    def otype(self, value):
        self.ft.set_otype(ord(value))

    @property
    def functions(self): 
        return self.ft.get_functions()
    @functions.setter
    def functions(self, value):
        self.ft.set_functions(value)

    @property
    def max_depth(self):   
        return self.ft.get_max_depth()
    @max_depth.setter
    def max_depth(self, value):
        self.ft.set_max_depth(value)

    @property
    def max_dim(self):  
        return self.ft.get_max_dim()
    @max_dim.setter
    def max_dim(self, value):
        self.ft.set_max_dim(value)

    @property
    def random_state(self): 
        return self.ft.get_random_state()
    @random_state.setter
    def random_state(self, value):
        self.ft.set_random_state(value)

    @property
    def erc(self):  
        return self.ft.get_erc()
    @erc.setter
    def erc(self, value):
        self.ft.set_erc(value)

    @property
    def obj(self): 
        return self.ft.get_obj()
    @obj.setter
    def obj(self, value):
        self.ft.set_obj(value)

    @property
    def shuffle(self):  
        return self.ft.get_shuffle()
    @shuffle.setter
    def shuffle(self, value):
        self.ft.set_shuffle(value)

    @property
    def split(self):  
        return self.ft.get_split()
    @split.setter
    def split(self, value):
        self.ft.set_split(value)

    @property
    def fb(self):
        return self.ft.get_fb()
    @fb.setter
    def fb(self, value):
        self.ft.set_fb(value)

    @property
    def scorer(self):
        return self.ft.get_scorer()
    @scorer.setter
    def scorer(self, value):
        self.ft.set_scorer(value)

    @property
    def feature_names(self):
        return self.ft.get_feature_names()
    @feature_names.setter
    def feature_names(self, value):
        self.ft.set_feature_names(value)

    @property
    def backprop(self):
        return self.ft.get_backprop()
    @backprop.setter
    def backprop(self, value):
        self.ft.set_backprop(value)

    @property
    def iters(self):
        return self.ft.get_iters()
    @iters.setter
    def iters(self, value):
        self.ft.set_iters(value)

    @property
    def lr(self):
        return self.ft.get_lr()
    @lr.setter
    def lr(self, value):
        self.ft.set_lr(value)

    @property
    def batch_size(self):
        return self.ft.get_batch_size()
    @batch_size.setter
    def batch_size(self, value):
        self.ft.set_batch_size(value)

    @property
    def n_threads(self):
        return self.ft.get_n_threads()
    @n_threads.setter
    def n_threads(self, value):
        self.ft.set_n_threads(value)

    @property
    def hillclimb(self):
        return self.ft.get_hillclimb()
    @hillclimb.setter
    def hillclimb(self, value):
        self.ft.set_hillclimb(value)

    @property
    def logfile(self):
        return self.ft.get_logfile()
    @logfile.setter
    def logfile(self, value):
        self.ft.set_logfile(value)

    @property
    def max_time(self):
        return self.ft.get_max_time()
    @max_time.setter
    def max_time(self, value):
        self.ft.set_max_time(value)

    @property
    def residual_xo(self):
        return self.ft.get_residual_xo()
    @residual_xo.setter
    def residual_xo(self, value):
        self.ft.set_residual_xo(value)

    @property
    def stagewise_xo(self):
        return self.ft.get_stagewise_xo()
    @stagewise_xo.setter
    def stagewise_xo(self, value):
        self.ft.set_stagewise_xo(value)

    @property
    def stagewise_xo_tol(self):
        return self.ft.get_stagewise_xo_tol()
    @stagewise_xo_tol.setter
    def stagewise_xo_tol(self, value):
        self.ft.set_stagewise_xo_tol(value)

    @property
    def softmax_norm(self):
        return self.ft.get_softmax_norm()
    @softmax_norm.setter
    def softmax_norm(self, value):
        self.ft.set_softmax_norm(value)

    @property
    def save_pop(self):
        return self.ft.get_save_pop()
    @save_pop.setter
    def save_pop(self, value):
        self.ft.set_save_pop(value)

    @property
    def normalize(self):
        return self.ft.get_normalize()
    @normalize.setter
    def normalize(self, value):
        self.ft.set_normalize(value)

    @property
    def val_from_arch(self):
        return self.ft.get_val_from_arch()
    @val_from_arch.setter
    def val_from_arch(self, value):
        self.ft.set_val_from_arch(value)

    @property
    def corr_delete_mutate(self):
        return self.ft.get_corr_delete_mutate()
    @corr_delete_mutate.setter
    def corr_delete_mutate(self, value):
        self.ft.set_corr_delete_mutate(value)

    @property
    def simplify(self):
        return self.ft.get_simplify()
    @simplify.setter
    def simplify(self, value):
        self.ft.set_simplify(value)

    @property
    def protected_groups(self):
        return self.ft.get_protected_groups()
    @protected_groups.setter
    def protected_groups(self, value):
        self.ft.set_protected_groups(value)

    @property
    def tune_initial(self):
        return self.ft.get_tune_initial()
    @tune_initial.setter
    def tune_initial(self, value):
        self.ft.set_tune_initial(value)

    @property
    def tune_final(self): 
        return self.ft.get_tune_final()
    @tune_final.setter
    def tune_final(self, value):
        self.ft.set_tune_final(value)

    @property
    def starting_pop(self):
        return self.ft.get_starting_pop()
    @starting_pop.setter
    def starting_pop(self, value):
        self.ft.set_starting_pop(value)

