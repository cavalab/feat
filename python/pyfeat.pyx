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
        void save(string filename)
        void load_best_ind(string filename)
        void load_population(string filename)
        bool fitted

cdef class PyFeat:
    """Feature Engineering Automation Tool

    Parameters
    ----------
    pop_size: int, optional (default: 100)
        Size of the population of models
    gens: int, optional (default: 100)
        Number of iterations to train for
    ml: str, optional (default: "LinearRidgeRegression")
        ML pairing. Choices: LinearRidgeRegression, Lasso, L1_LR, L2_LR
    classification: boolean, optional (default: False)
        Whether to do classification instead of regression. 
    verbosity: int, optional (default: 0)
        How much to print out (0, 1, 2)
    max_stall: int, optional (default: 0)
        How many generations to continue after the validation loss has
        stalled. If 0, not used.
    sel: str, optional (default: "lexicase")
        Selection algorithm to use.   
    surv: str, optional (default: "nsga2")
        Survival algorithm to use. 
    cross_rate: float, optional (default: 0.5)
        How often to do crossover for variation versus mutation. 
    root_xo_rate: float, optional (default: 0.5)
        When performing crossover, how often to choose from the roots of 
        the trees, rather than within the tree. Root crossover essentially
        swaps features in models.
    otype: string, optional (default: 'a')
        Feature output types:
        'a': all
        'b': boolean only
        'f': floating point only
    functions: string, optional (default: "")
        What operators to use to build features. If functions="", all the
        available functions are used. 
    max_depth: int, optional (default: 3)
        Maximum depth of a feature's tree representation.
    max_dim: int, optional (default: 10)
        Maximum dimension of a model. The dimension of a model is how many
        independent features it has. Controls the number of trees in each 
        individual.
    random_state: int, optional (default: 0)
        Random seed.
    erc: boolean, optional (default: False)
        If true, ephemeral random constants are included as nodes in trees.
    obj: str, optional (default: "fitness,complexity")
        Objectives to use for multi-objective optimization.
    shuffle: boolean, optional (default: True)
        Whether to shuffle the training data before beginning training.
    split: float, optional (default: 0.75)
        The internal fraction of training data to use. The validation fold
        is then 1-split fraction of the data. 
    fb: float, optional (default: 0.5)
        Controls the amount of feedback from the ML weights used during 
        variation. Higher values make variation less random.
    scorer: str, optional (default: '')
        Scoring function to use internally. 
    feature_names: str, optional (default: '')
        Optionally provide comma-separated feature names. Should be equal
        to the number of features in your data. This will be set 
        automatically if a Pandas dataframe is passed to fit(). 
    backprop: boolean, optional (default: False)
        Perform gradient descent on feature weights using backpropagation.
    iters: int, optional (default: 10)
        Controls the number of iterations of backprop as well as 
        hillclimbing for learning weights.
    lr: float, optional (default: 0.1)
        Learning rate used for gradient descent. This the initial rate, 
        and is scheduled to decrease exponentially with generations. 
    batch_size: int, optional (default: 0)
        Number of samples to train on each generation. 0 means train on 
        all the samples.
    n_jobs: int, optional (default: 0)
        Number of parallel threads to use. If 0, this will be 
        automatically determined by OMP. 
    hillclimb: boolean, optional (default: False)
        Applies stochastic hillclimbing to feature weights. 
    logfile: str, optional (default: "")
        If specified, spits statistics into a logfile. "" means don't log.
    max_time: int, optional (default: -1)
        Maximum time terminational criterion in seconds. If -1, not used.
    residual_xo: boolean, optional (default: False)
        Use residual crossover. 
    stagewise_xo: boolean, optional (default: False)
        Use stagewise crossover.
    stagewise_xo_tol: boolean, optional (default:False)
        Terminates stagewise crossover based on an error value rather than
        dimensionality. 
    softmax_norm: boolean, optional (default: False)
        Uses softmax normalization of probabilities of variation across
        the features. 
    save_pop: int, optional (default: 0)
        Prints the population of models. 0: don't print; 1: print final 
        population; 2: print every generation. 
    normalize: boolean, optional (default: True)
        Normalizes the floating point input variables using z-scores. 
    val_from_arch: boolean, optional (default: True)
        Validates the final model using the archive rather than the whole 
        population.
    corr_delete_mutate: boolean, optional (default: False)
        Replaces root deletion mutation with a deterministic deletion 
        operator that deletes the feature with highest collinearity. 
    simplify: float, optional (default: 0)
        Runs post-run simplification to try to shrink the final model 
        without changing its output more than the simplify tolerance.
        This tolerance is the norm of the difference in outputs, divided
        by the norm of the output. If simplify=0, it is ignored. 
    protected_groups: list, optional (default: [])
        Defines protected attributes in the data. Uses for adding 
        fairness constraints. 
    tune_initial: boolean, optional (default: False)
        Tune the initial linear model's penalization parameter. 
    tune_final: boolean, optional (default: True)
        Tune the final linear model's penalization parameter. 
    starting_pop: str, optional (default: "")
        Provide a starting pop in json format. 
    """
    cdef Feat ft  # hold a c++ instance which we're wrapping
    def __cinit__(self, 
                  int pop_size=100, 
                  int gens=100, 
                  string ml= b"LinearRidgeRegression", 
                  bool classification=False, 
                  int verbosity=0, 
                  int max_stall=0, 
                  string sel="lexicase", 
                  string surv="nsga2", 
                  float cross_rate=0.5, 
                  float root_xo_rate=0.5, 
                  char otype='a', 
                  string functions="", 
                  unsigned int max_depth=3, 
                  unsigned int max_dim=10, 
                  int random_state=0, 
                  bool erc= False , 
                  string obj="fitness,complexity", 
                  bool shuffle=True, 
                  float split=0.75, 
                  float fb=0.5, 
                  string scorer='', 
                  string feature_names="", 
                  bool backprop=False, 
                  int iters=10, 
                  float lr=0.1, 
                  int batch_size=0, 
                  int n_jobs=0, 
                  bool hillclimb=False, 
                  string logfile="", 
                  int max_time=-1, 
                  bool residual_xo=False, 
                  bool stagewise_xo=False, 
                  bool stagewise_xo_tol=False, 
                  bool softmax_norm=False, 
                  int save_pop=0, 
                  bool normalize=True, 
                  bool val_from_arch=True, 
                  bool corr_delete_mutate=False, 
                  float simplify=0.0, 
                  string protected_groups="", 
                  bool tune_initial=False, 
                  bool tune_final=True, 
                  string starting_pop=""
                 ):
        # if len(otype) > 1:
        #     print('otype len is >1:',otype)
        # cdef char otype_char = ord(otype)

        self.ft = Feat(pop_size,gens,ml,classification, verbosity,max_stall,
                       sel, surv, cross_rate, root_xo_rate, otype, 
                       functions, max_depth, max_dim, random_state, erc, obj, 
                       shuffle, split, fb, scorer, feature_names, backprop, 
                       iters, lr, batch_size, n_jobs, hillclimb, logfile, 
                       max_time, residual_xo, stagewise_xo, stagewise_xo_tol, 
                       softmax_norm, save_pop, normalize, val_from_arch, 
                       corr_delete_mutate, simplify, protected_groups, 
                       tune_initial, tune_final, starting_pop)

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

    def get_archive(self,justfront=False):
        """Returns all the final representation equations in the archive"""
        archive = []
        str_arc = self.ft.get_archive(justfront).decode()
        for model in str_arc.splitlines():
            archive.append( json.loads(model))
        return archive

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
        
    def get_archive_size(self):
        return self.ft.get_archive_size()

    def load(self, filename):
        self.ft.load(str(filename).encode())
        return self

    def save(self, filename):
        self.ft.save(str(filename).encode())

    def load_best_ind(self, filename):
        self.ft.load_best_ind(str(filename).encode())

    def _load_population(self, filename):
        self.ft.load_population(filename)

    # def _get_params():
    #     return json.loads(self.ft.get_params())

    ###########################################################################
    # decorated property setters and getters
    ###########################################################################
    @property
    def stats_(self):
        return json.loads(self.ft.get_stats())

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
        cdef char otype = self.ft.get_otype()
        return otype
        # return str(self.ft.get_otype()).encode()
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
    def n_jobs(self):
        return self.ft.get_n_jobs()
    @n_jobs.setter
    def n_jobs(self, value):
        self.ft.set_n_jobs(value)

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

    @property
    def fitted_(self):
        return self.ft.fitted

    def get_params(self, deep=False):
        return {
                'pop_size': self.pop_size, 
                'gens': self.gens, 
                'ml': self.ml,
                'classification': self.classification,
                'verbosity': self.verbosity,
                'max_stall': self.max_stall,
                'sel': self.sel,
                'surv': self.surv,
                'cross_rate': self.cross_rate,
                'root_xo_rate': self.root_xo_rate,
                'otype': self.otype,
                'functions': self.functions,
                'max_depth': self.max_depth,
                'max_dim': self.max_dim,
                'random_state': self.random_state,
                'erc': self.erc,
                'obj': self.obj,
                'shuffle': self.shuffle,
                'split': self.split,
                'fb': self.fb,
                'scorer': self.scorer,
                'feature_names': self.feature_names,
                'backprop': self.backprop,
                'iters': self.iters,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'n_jobs': self.n_jobs,
                'hillclimb': self.hillclimb,
                'logfile': self.logfile,
                'max_time': self.max_time,
                'residual_xo': self.residual_xo,
                'stagewise_xo': self.stagewise_xo,
                'stagewise_xo_tol': self.stagewise_xo_tol,
                'softmax_norm': self.softmax_norm,
                'save_pop': self.save_pop,
                'normalize': self.normalize,
                'val_from_arch': self.val_from_arch,
                'corr_delete_mutate': self.corr_delete_mutate,
                'simplify': self.simplify,
                'protected_groups': self.protected_groups,
                'tune_initial': self.tune_initial,
                'tune_final': self.tune_final,
                'starting_pop': self.starting_pop,
                }
