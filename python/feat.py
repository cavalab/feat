# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

import argparse
from versionstr import __version__
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import pyfeat as pyfeat
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pdb

class Feat(BaseEstimator):
    def __init__(self, pop_size=100,  gens=100,  ml = "LinearRidgeRegression", 
                classification=False,  verbosity=0,  max_stall=0,
                sel ="lexicase",  surv ="nsga2",  cross_rate=0.5, 
                root_xo_rate=0.5, otype ='a',  functions ="", 
                max_depth=3,   max_dim=10,  random_state=0, 
                erc = False,  obj ="fitness,complexity", shuffle=True,  
                split=0.75,  fb=0.5, scorer ='',feature_names="", 
                backprop=False, iters=10, lr=0.1, batch_size=0, 
                n_threads=0, hillclimb=False, logfile="", max_time=-1, 
                residual_xo=False, stagewise_xo=False, 
                stagewise_xo_tol=False, softmax_norm=False, print_pop=0, 
                normalize=True, val_from_arch=True, corr_delete_mutate=False, 
                simplify=False,protected_groups=[],
                tune_initial=False, tune_final=True):
        """Feat uses GP to find a data representation that improves the 
        performance of a given ML method.

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






        
        """
        self.pop_size = pop_size
        self.gens = gens
        self.ml = ml.encode() if( isinstance(ml,str) )  else ml

        self.classification = classification
        self.verbosity = verbosity
        self.max_stall = max_stall
        self.sel = sel.encode() if( isinstance(sel,str) )  else sel
        self.surv = surv.encode() if( isinstance(surv,str) )  else surv
        self.cross_rate = cross_rate
        self.root_xo_rate = root_xo_rate
        self.otype = otype.encode() if( isinstance(otype,str) )  else otype
        self.functions = (functions.encode() if( isinstance(functions,str) )  
                else functions)
        self.max_depth = max_depth
        self.max_dim = max_dim
        self.random_state = int(random_state)
        self.erc = erc      
        self.obj = obj.encode() if( isinstance(obj,str) )  else obj
        self.shuffle = shuffle
        self.split = split
        self.fb = fb
        self.scorer = scorer.encode() if( isinstance(scorer,str) )  else scorer
        self.feature_names = (feature_names.encode() 
                if isinstance(feature_names,str) else feature_names )
        self.backprop = bool(backprop)
        self.iters = int(iters)
        self.lr = float(lr)
        self.batch_size= int(batch_size)
        self.n_threads = int(n_threads)
        self.hillclimb= bool(hillclimb) 
        self.logfile = logfile.encode() if isinstance(logfile,str) else logfile
        self.max_time = max_time
        self.residual_xo = residual_xo
        self.stagewise_xo = stagewise_xo
        self.stagewise_xo_tol = stagewise_xo_tol
        self.softmax_norm = softmax_norm
        self.print_pop = print_pop
        self.normalize = normalize
        self.val_from_arch = val_from_arch
        self.corr_delete_mutate = corr_delete_mutate
        self.simplify = simplify
        self.protected_groups = ','.join(
                [str(int(pg)) for pg in protected_groups]).encode()
        self.tune_initial = tune_initial
        self.tune_final = tune_final
        self._pyfeat=None
        self.stats = {}
        self.__version__ = __version__

    def _init_pyfeat(self):
        """set up pyfeat glue class object"""
        self._pyfeat = pyfeat.PyFeat( self.pop_size,  self.gens,  self.ml, 
                self.classification,  self.verbosity,  self.max_stall,
                self.sel,  self.surv,  self.cross_rate, self.root_xo_rate,
                self.otype,  self.functions, 
                self.max_depth,   self.max_dim,  self.random_state, 
                self.erc,  
                self.obj, 
                self.shuffle,  
                self.split,  
                self.fb,
                self.scorer,
                self.feature_names,
                self.backprop,
                self.iters,
                self.lr,
                self.batch_size,
                self.n_threads,
                self.hillclimb,
                self.logfile,
                self.max_time,
                self.residual_xo,
                self.stagewise_xo,
                self.stagewise_xo_tol,
                self.softmax_norm,
                self.print_pop,
                self.normalize,
                self.val_from_arch,
                self.corr_delete_mutate,
                self.simplify,
                self.protected_groups,
                self.tune_initial,
                self.tune_final)
   

    def fit(self,X,y,zfile=None,zids=None):
        """Fit a model."""    
        # if type(X).__name__ == 'DataFrame':
        #     if len(list(X.columns)) == X.shape[1]:
        #         self.feature_names = ','.join(X.columns).encode()
        #     X = X.values
        # if type(y).__name__ in ['DataFrame','Series']:
        #     y = y.values
        X = self._clean(X, set_feature_names=True)
        y = self._clean(y)

        self._init_pyfeat()   
        
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            self._pyfeat.fit_with_z(X,y,zfile,zids)
        else:
            self._pyfeat.fit(X,y)

        self.update_stats()

        return self

    def predict(self,X,zfile=None,zids=None):
        """Predict on X."""
        X = self._clean(X)
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.predict_with_z(X,zfile,zids)
        else:
            return self._pyfeat.predict(X)

    def predict_archive(self,X,zfile=None,zids=None):
        """Returns a list of dictionary predictions for all models."""
        X = self._clean(X)

        if zfile:
            raise ImplementationError('longitudinal not implemented')
            return 1

        archive = self.get_archive(justfront=False)
        preds = []
        for ind in archive:
            tmp = {}
            tmp['id'] = ind['id']
            tmp['y_pred'] = self._pyfeat.predict_archive(ind['id'], X) 
            preds.append(tmp)

        return preds

    def predict_proba_archive(self,X,zfile=None,zids=None):
        """Returns a dictionary of prediction probabilities for all models."""
        X = self._clean(X)
        if zfile:
            raise ImplementationError('longitudinal not implemented')
            # zfile = zfile.encode() if isinstance(zfile,str) else zfile
            # return self._pyfeat.predict_with_z(X,zfile,zids)
            return 1

        archive = self.get_archive()
        probs = []
        for ind in archive:
            tmp = {}
            tmp['id'] = ind['id']
            tmp['y_proba'] = self._pyfeat.predict_proba_archive(ind['id'], X)
            probs.append(tmp)

        return probs

    def predict_proba(self,X,zfile=None,zids=None):
        """Return probabilities of predictions for data X"""
        X = self._clean(X)
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            tmp = self._pyfeat.predict_proba_with_z(X,zfile,zids)
        else:
            tmp = self._pyfeat.predict_proba(X)
        
        if len(tmp.shape)<2:
                tmp  = np.vstack((1-tmp,tmp)).transpose()
        return tmp         


    def transform(self,X,zfile=None,zids=None):
        """Return the representation's transformation of X"""
        X = self._clean(X)
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.transform_with_z(X,zfile,zids)
        else:
            return self._pyfeat.transform(X)

    def fit_predict(self,X,y):
        """Convenience method that runs fit(X,y) then predict(X)"""
        # X, _ = self._clean(X)
        # self._init_pyfeat()    
        # result = self._pyfeat.fit_predict(X,y)
        # self.update_stats()
        self.fit(X,y)
        result = self.predict(X)
        return result

    def fit_transform(self,X,y):
        """Convenience method that runs fit(X,y) then transform(X)"""
        self.fit(X,y)
        result = self.transform(X)
        # self._init_pyfeat() 
        # result = self._pyfeat.fit_transform(X, y)
        # self.update_stats()   
        return result

    def score(self,X,y,zfile=None,zids=None):
        """Returns a score for the predictions of Feat on X versus true 
        labels y""" 
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            yhat = self._pyfeat.predict_with_z(X,zfile,zids).flatten()
        else:
            yhat = self.predict(X).flatten()
        if ( self.classification ):
            return log_loss(y,yhat, labels=y)
        else:
            return mse(y,yhat)

    def get_model(self):
        """Returns a string with the set of equations and weights in the final 
        representation"""
        return self._pyfeat.get_model()

    def get_representation(self):
        """Returns a string with the final representation"""
        return self._pyfeat.get_representation()

    def _typify(self, x):
        """Tries to typecast argument to a numeric type."""
        try:
            return float(x)
        except:
            try:
                return int(x)
            except:
                return x
            
    def get_archive(self,justfront=False):
        """Returns all the final representation equations in the archive"""
        str_arc = self._pyfeat.get_archive(justfront)
        # store archive data from string
        archive=[]
        index = {}
        for i,s in enumerate(str_arc.split('\n')):
            if i == 0:
                for j,key in enumerate(s.split('\t')):
                    index[j] = key
            else:
                ind= {}
                for k,val in enumerate(s.split('\t')):
                    if ',' in val:
                        ind[index[k]] = []
                        for el in val.split(','):
                            ind[index[k]].append(self._typify(el))
                        continue
                    ind[index[k]] = self._typify(val)
                archive.append(ind)
        return archive

    def get_archive_size(self):
        return self._pyfeat.get_archive_size()

    def get_coefs(self):
        """Returns the coefficients assocated with each feature in the 
        representation"""
        return self._pyfeat.get_coefs()

    def get_dim(self):
        """Returns the dimensionality of the final representation"""
        return self._pyfeat.get_dim()

    def get_n_params(self):
        """Returns the number of parameters in the final representation"""
        return self._pyfeat.get_n_params()

    def get_complexity(self):
        """Returns the complexity of the final representation"""
        return self._pyfeat.get_complexity()

    def get_n_nodes(self):
        """Returns the number of nodes in the final representation"""
        return self._pyfeat.get_n_nodes()

    def update_stats(self):
        """updates the statistics of the run"""
        self.stats["gens"] = self._pyfeat.get_gens()
        self.stats["time"] = self._pyfeat.get_timers()
        self.stats["min_losses"] = self._pyfeat.get_min_losses()
        self.stats["min_losses_val"] = self._pyfeat.get_min_losses_val()
        self.stats["med_scores"] = self._pyfeat.get_med_scores()
        self.stats["med_loss_vals"] = self._pyfeat.get_med_loss_vals()
        self.stats["med_size"] = self._pyfeat.get_med_size()
        self.stats["med_complexity"] = self._pyfeat.get_med_complexities()
        self.stats["med_num_params"] = self._pyfeat.get_med_num_params()
        self.stats["med_dim"] = self._pyfeat.get_med_dim()
        self.feature_importances_ = self.get_coefs()

    def _clean(self, x, set_feature_names=False):
        """Converts dataframe to array, optionally returning feature names"""
        feature_names = ''.encode()
        if type(x).__name__ == 'DataFrame':
            if set_feature_names and len(list(x.columns)) == x.shape[1]:
                self.feature_names = ','.join(x.columns).encode()
                return x.values
            else:
                return x.values
        elif type(x).__name__ == 'Series':
            return x.values
        else:
            assert(type(x).__name__ == 'ndarray')
            return x
