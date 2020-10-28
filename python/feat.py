# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

import argparse
from versionstr import __version__
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd
from pyfeat import PyFeat
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss
import pdb
import json


class Feat(PyFeat, BaseEstimator):

    def __init__(self, 
                 pop_size=100, 
                 gens=100, 
                 ml= "LinearRidgeRegression", 
                 classification=False, 
                 verbosity=0, 
                 max_stall=0, 
                 sel="lexicase", 
                 surv="nsga2", 
                 cross_rate=0.5, 
                 root_xo_rate=0.5, 
                 otype='a', 
                 functions="", 
                 max_depth=3, 
                 max_dim=10, 
                 random_state=0, 
                 erc= False , 
                 obj="fitness,complexity", 
                 shuffle=True, 
                 split=0.75, 
                 fb=0.5, 
                 scorer='', 
                 feature_names="", 
                 backprop=False, 
                 iters=10, 
                 lr=0.1, 
                 batch_size=0, 
                 n_jobs=0, 
                 hillclimb=False, 
                 logfile="", 
                 max_time=-1, 
                 residual_xo=False, 
                 stagewise_xo=False, 
                 stagewise_xo_tol=False, 
                 softmax_norm=False, 
                 save_pop=0, 
                 normalize=True, 
                 val_from_arch=True, 
                 corr_delete_mutate=False, 
                 simplify=0.0, 
                 protected_groups="", 
                 tune_initial=False, 
                 tune_final=True, 
                 starting_pop="",
                )
        self.pop_size=pop_size
        self.gens=gens
        self.ml=ml
        self.classification=classification
        self.verbosity=verbosity
        self.max_stall=max_stall
        self.sel=sel
        self.surv=surv
        self.cross_rate=cross_rate
        self.root_xo_rate=root_xo_rate
        self.otype=otype
        self.functions=functions
        self.max_depth=max_depth
        self.max_dim=max_dim
        self.random_state=random_state
        self.erc=erc
        self.obj=obj
        self.shuffle=shuffle
        self.split=split
        self.fb=fb
        self.scorer=scorer
        self.feature_names=feature_names
        self.backprop=backprop
        self.iters=iters
        self.lr=lr
        self.batch_size=batch_size
        self.n_jobs=n_jobs
        self.hillclimb=hillclimb
        self.logfile=logfile
        self.max_time=max_time
        self.residual_xo=residual_xo
        self.stagewise_xo=stagewise_xo
        self.stagewise_xo_tol=stagewise_xo_tol
        self.softmax_norm=softmax_norm
        self.save_pop=save_pop
        self.normalize=normalize
        self.val_from_arch=val_from_arch
        self.corr_delete_mutate=corr_delete_mutate
        self.simplify=simplify
        self.protected_groups=protected_groups
        self.tune_initial=tune_initial
        self.tune_final=tune_final
        self.starting_pop=starting_pop

        self._set_params(self.get_params())
   
    def set_params(self, params):
        for k,v in params.items():
            setattr(self,k,v)

    def load(self, filename):
        self._load(filename)
        self.set_params({k[1:]:v for k,v in self._get_params().items()})

    def get_params(self, deep=False):
        return self.__dict__

    def fit(self,X,y,zfile=None,zids=None):
        """Fit a model."""    
        self._set_params({'_'+k:v for k,v in self.get_params().items()})

        X = self._clean(X, set_feature_names=True)
        y = self._clean(y)

        if zfile:
            self._fit_with_z(X,y,zfile,zids)
        else:
            self._fit(X,y)

        return self

    def predict(self,X,zfile=None,zids=None):
        """Predict on X."""
        if not self.fitted_:
            raise ValueError("Call fit before calling predict.")


        X = self._clean(X)
        if zfile:
            return self._predict_with_z(X,zfile,zids)
        else:
            return self._predict(X)

    def predict_archive(self,X,zfile=None,zids=None):
        """Returns a list of dictionary predictions for all models."""
        if not self.fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._clean(X)

        if zfile:
            raise ImplementationError('longitudinal not implemented')
            return 1

        archive = self.get_archive(justfront=False)
        preds = []
        for ind in archive:
            tmp = {}
            tmp['id'] = ind['id']
            tmp['y_pred'] = self._predict_archive(ind['id'], X) 
            preds.append(tmp)

        return preds


    def transform(self,X,zfile=None,zids=None):
        """Return the representation's transformation of X"""
        if not self.fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._clean(X)
        if zfile:
            return self._transform_with_z(X,zfile,zids)
        else:
            return self._transform(X)

    def fit_predict(self,X,y):
        """Convenience method that runs fit(X,y) then predict(X)"""
        self.fit(X,y)
        result = self.predict(X)
        return result

    def fit_transform(self,X,y):
        """Convenience method that runs fit(X,y) then transform(X)"""
        self.fit(X,y)
        result = self.transform(X)
        return result

    def score(self,X,y,zfile=None,zids=None):
        """Returns a score for the predictions of Feat on X versus true 
        labels y""" 
        yhat = self.predict(X,zfile,zids).flatten()
        if ( self.classification ):
            return log_loss(y,yhat, labels=y)
        else:
            return mse(y,yhat)

    def _clean(self, x, set_feature_names=False):
        """Converts dataframe to array, optionally returning feature names"""
        feature_names = ''
        if type(x).__name__ == 'DataFrame':
            if set_feature_names and len(list(x.columns)) == x.shape[1]:
                self.feature_names = ','.join(x.columns)
                return x.values
            else:
                return x.values
        elif type(x).__name__ == 'Series':
            return x.values
        else:
            assert(type(x).__name__ == 'ndarray')
            return x

# class FeatRegressor(Feat, RegressorMixin):
    # def __new__(cls, **kwargs):
    #     kwargs['classification'] = False
    #     if 'ml' not in kwargs: kwargs['ml'] = b'Ridge'
    #     print('FeatRegressor.new', kwargs)
    #     return super().__new__(cls, **kwargs)
    # def fit(self,X,y,zfile=None,zids=None):
    #     self.classification = False
    #     Feat.fit(self,X,y,zfile,zids)

# class FeatClassifier(Feat,ClassifierMixin):
    # def __new__(cls, **kwargs):
    #     kwargs['classification'] = True
    #     if 'ml' not in kwargs: kwargs['ml'] = b'L2_LR'
    #     print('FeatClassifier.new', kwargs)
    #     return super().__new__(cls, **kwargs)

    # def fit(self,X,y,zfile=None,zids=None):
    #     self.classification = True
    #     # check ml is classifier
    #     if self.ml == 'Ridge': 
    #         self.ml = b'LR'
    #     Feat.fit(self, X, y)

        

    def predict_proba(self,X,zfile=None,zids=None):
        """Return probabilities of predictions for data X"""
        if not self.fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._clean(X)
        if zfile:
            tmp = self._predict_proba_with_z(X,zfile,zids)
        else:
            tmp = self._predict_proba(X)
        
        if len(tmp.shape)<2:
                tmp  = np.vstack((1-tmp,tmp)).transpose()
        return tmp         

    def predict_proba_archive(self,X,zfile=None,zids=None):
        """Returns a dictionary of prediction probabilities for all models."""
        if not self.fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._clean(X)
        if zfile:
            raise ImplementationError('longitudinal not implemented')
            # return self._predict_with_z(X,zfile,zids)
            return 1

        archive = self.get_archive()
        probs = []
        for ind in archive:
            tmp = {}
            tmp['id'] = ind['id']
            tmp['y_proba'] = self._predict_proba_archive(ind['id'], X)
            probs.append(tmp)

        return probs
    ###########################################################################
    # decorated property setters and getters
    ############################################################################
    #@property
    #def stats_(self):
    #    return self._stats_ 

    #@property
    #def pop_size(self):  
    #    return self._pop_size
    #@pop_size.setter
    #def pop_size(self, value):
    #    self._pop_size = value

    #@property
    #def gens(self):  
    #    return self._gens
    #@gens.setter
    #def gens(self, value):
    #    self._gens = value

    #@property
    #def ml(self): 
    #    return self._ml
    #@ml.setter
    #def ml(self, value):
    #    self._ml = value 

    #@property
    #def classification(self):  
    #    return self._classification
    #@classification.setter
    #def classification(self, value):
    #    self._classification = value

    #@property
    #def verbosity(self):  
    #    return self._verbosity
    #@verbosity.setter
    #def verbosity(self, value):
    #    self._verbosity = value

    #@property
    #def max_stall(self):
    #    return self._max_stall
    #@max_stall.setter
    #def max_stall(self, value):
    #    self._max_stall = value

    #@property
    #def sel(self):  
    #    return self._sel
    #@sel.setter
    #def sel(self, value):
    #    self._sel = value

    #@property
    #def surv(self):  
    #    return self._surv
    #@surv.setter
    #def surv(self, value):
    #    self._surv = value

    #@property
    #def cross_rate(self): 
    #    return self._cross_rate
    #@cross_rate.setter
    #def cross_rate(self, value):
    #    self._cross_rate = value

    #@property
    #def root_xo_rate(self):
    #    return self._root_xo_rate
    #@root_xo_rate.setter
    #def root_xo_rate(self, value):
    #    self._root_xo_rate = value

    #@property
    #def otype(self):  
    #    return self._otype
    #@otype.setter
    #def otype(self, value):
    #    self._otype = value

    #@property
    #def functions(self): 
    #    return self._functions
    #@functions.setter
    #def functions(self, value):
    #    self._functions = value

    #@property
    #def max_depth(self):   
    #    return self._max_depth
    #@max_depth.setter
    #def max_depth(self, value):
    #    self._max_depth = value

    #@property
    #def max_dim(self):  
    #    return self._max_dim
    #@max_dim.setter
    #def max_dim(self, value):
    #    self._max_dim = value

    #@property
    #def random_state(self): 
    #    return self._random_state
    #@random_state.setter
    #def random_state(self, value):
    #    self._random_state = value

    #@property
    #def erc(self):  
    #    return self._erc
    #@erc.setter
    #def erc(self, value):
    #    self._erc = value

    #@property
    #def obj(self): 
    #    return self._obj
    #@obj.setter
    #def obj(self, value):
    #    self._obj = value

    #@property
    #def shuffle(self):  
    #    return self._shuffle
    #@shuffle.setter
    #def shuffle(self, value):
    #    self._shuffle = value

    #@property
    #def split(self):  
    #    return self._split
    #@split.setter
    #def split(self, value):
    #    self._split = value

    #@property
    #def fb(self):
    #    return self._fb
    #@fb.setter
    #def fb(self, value):
    #    self._fb = value

    #@property
    #def scorer(self):
    #    return self._scorer
    #@scorer.setter
    #def scorer(self, value):
    #    self._scorer = value

    #@property
    #def feature_names(self):
    #    return self._feature_names
    #@feature_names.setter
    #def feature_names(self, value):
    #    self._feature_names = value

    #@property
    #def backprop(self):
    #    return self._backprop
    #@backprop.setter
    #def backprop(self, value):
    #    self._backprop = value

    #@property
    #def iters(self):
    #    return self._iters
    #@iters.setter
    #def iters(self, value):
    #    self._iters = value

    #@property
    #def lr(self):
    #    return self._lr
    #@lr.setter
    #def lr(self, value):
    #    self._lr = value

    #@property
    #def batch_size(self):
    #    return self._batch_size
    #@batch_size.setter
    #def batch_size(self, value):
    #    self._batch_size = value

    #@property
    #def n_jobs(self):
    #    return self._n_jobs
    #@n_jobs.setter
    #def n_jobs(self, value):
    #    self._n_jobs = value

    #@property
    #def hillclimb(self):
    #    return self._hillclimb
    #@hillclimb.setter
    #def hillclimb(self, value):
    #    self._hillclimb = value

    #@property
    #def logfile(self):
    #    return self._logfile
    #@logfile.setter
    #def logfile(self, value):
    #    self._logfile = value

    #@property
    #def max_time(self):
    #    return self._max_time
    #@max_time.setter
    #def max_time(self, value):
    #    self._max_time = value

    #@property
    #def residual_xo(self):
    #    return self._residual_xo
    #@residual_xo.setter
    #def residual_xo(self, value):
    #    self._residual_xo = value

    #@property
    #def stagewise_xo(self):
    #    return self._stagewise_xo
    #@stagewise_xo.setter
    #def stagewise_xo(self, value):
    #    self._stagewise_xo = value

    #@property
    #def stagewise_xo_tol(self):
    #    return self._stagewise_xo_tol
    #@stagewise_xo_tol.setter
    #def stagewise_xo_tol(self, value):
    #    self._stagewise_xo_tol = value

    #@property
    #def softmax_norm(self):
    #    return self._softmax_norm
    #@softmax_norm.setter
    #def softmax_norm(self, value):
    #    self._softmax_norm = value

    #@property
    #def save_pop(self):
    #    return self._save_pop
    #@save_pop.setter
    #def save_pop(self, value):
    #    self._save_pop = value

    #@property
    #def normalize(self):
    #    return self._normalize
    #@normalize.setter
    #def normalize(self, value):
    #    self._normalize = value

    #@property
    #def val_from_arch(self):
    #    return self._val_from_arch
    #@val_from_arch.setter
    #def val_from_arch(self, value):
    #    self._val_from_arch = value

    #@property
    #def corr_delete_mutate(self):
    #    return self._corr_delete_mutate
    #@corr_delete_mutate.setter
    #def corr_delete_mutate(self, value):
    #    self._corr_delete_mutate = value

    #@property
    #def simplify(self):
    #    return self._simplify
    #@simplify.setter
    #def simplify(self, value):
    #    self._simplify = value

    #@property
    #def protected_groups(self):
    #    return self._protected_groups
    #@protected_groups.setter
    #def protected_groups(self, value):
    #    self._protected_groups = value

    #@property
    #def tune_initial(self):
    #    return self._tune_initial
    #@tune_initial.setter
    #def tune_initial(self, value):
    #    self._tune_initial = value

    #@property
    #def tune_final(self): 
    #    return self._tune_final
    #@tune_final.setter
    #def tune_final(self, value):
    #    self._tune_final = value

    #@property
    #def starting_pop(self):
    #    return self._starting_pop
    #@starting_pop.setter
    #def starting_pop(self, value):
    #    self._starting_pop = value

    #@property
    #def fitted_(self):
    #    return self._fitted_
