# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

import argparse
#from ._version import __version__

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import pyfeat
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pdb

class Feat(BaseEstimator):
    """Feat uses GP to find a data representation that improves the performance of a given ML
    method."""
    def __init__(self, pop_size=100,  gens=100,  ml = "LinearRidgeRegression", 
                classification=False,  verbosity=0,  max_stall=0,
                sel ="lexicase",  surv ="nsga2",  cross_rate=0.5,
                otype ='a',  functions ="", 
                max_depth=3,   max_dim=10,  random_state=0, 
                erc = False,  obj ="fitness,complexity", shuffle=True,  split=0.75,  fb=0.5,
                scorer ='',feature_names="", backprop=False, iters=10, lr=0.1, batch_size=100, 
                n_threads=0, hillclimb=False, logfile="Feat.log", max_time=-1, use_batch=False, 
                semantic_xo=False, print_pop=0):
        self.pop_size = pop_size
        self.gens = gens
        self.ml = ml.encode() if( isinstance(ml,str) )  else ml

        self.classification = classification
        self.verbosity = verbosity
        self.max_stall = max_stall
        self.sel = sel.encode() if( isinstance(sel,str) )  else sel
        self.surv = surv.encode() if( isinstance(surv,str) )  else surv
        self.cross_rate = cross_rate
        self.otype = otype.encode() if( isinstance(otype,str) )  else otype
        self.functions = functions.encode() if( isinstance(functions,str) )  else functions
        self.max_depth = max_depth
        self.max_dim = max_dim
        self.random_state = int(random_state)
        self.erc = erc      
        self.obj = obj.encode() if( isinstance(obj,str) )  else obj
        self.shuffle = shuffle
        self.split = split
        self.fb = fb
        self.scorer = scorer.encode() if( isinstance(scorer,str) )  else scorer
        self.feature_names = (feature_names.encode() if isinstance(feature_names,str) 
                                                     else feature_names )
        self.backprop = bool(backprop)
        self.iters = int(iters)
        self.lr = float(lr)
        if batch_size:
            self.batch_size= int(batch_size)
        else:
            print('batch_size is None for some reason')
            self.batch_size = 100

        self.n_threads = int(n_threads)
        self.hillclimb= bool(hillclimb) 
        self.logfile = logfile.encode() if isinstance(logfile,str) else logfile
        self.max_time = max_time
        self.use_batch = use_batch
        self.semantic_xo = semantic_xo
        self.print_pop = print_pop
        # if self.verbosity>0:
        #print('self.__dict__: ' , self.__dict__)
        self._pyfeat=None
        self.stats = {}

    def _init_pyfeat(self):
        """set up pyfeat glue class object"""
        self._pyfeat = pyfeat.PyFeat( self.pop_size,  self.gens,  self.ml, 
                self.classification,  self.verbosity,  self.max_stall,
                self.sel,  self.surv,  self.cross_rate,
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
                self.use_batch,
                self.semantic_xo,
                self.print_pop)

        self.stats["gens"] = self.get_gens()
        self.stats["time"] = self.get_timers()
        self.stats["best_scores"] = self.get_best_scores()
        self.stats["best_score_vals"] = self.get_best_score_vals()
        self.stats["med_scores"] = self.get_med_scores()
        self.stats["med_loss_vals"] = self.get_med_loss_vals()
        self.stats["med_size"] = self.get_med_size()
        self.stats["med_complexity"] = self.get_med_complexities()
        self.stats["med_num_params"] = self.get_med_num_params()
        self.stats["med_dim"] = self.get_med_dim()
   
    def fit(self,X,y,zfile=None,zids=None):
        """Fit a model."""    
        if type(X).__name__ == 'DataFrame':
            if len(list(X.columns)) == X.shape[1]:
                self.feature_names = ','.join(X.columns).encode()
            X = X.values
        if type(y).__name__ in ['DataFrame','Series']:
            y = y.values

        self._init_pyfeat()   
        
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            self._pyfeat.fit_with_z(X,y,zfile,zids)
        else:
            self._pyfeat.fit(X,y)

        return self

    def predict(self,X,zfile=None,zids=None):
        """Predict on X."""
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.predict_with_z(X,zfile,zids)
        else:
            return self._pyfeat.predict(X)

    def predict_proba(self,X,zfile=None,zids=None):
        """Return probabilities of predictions for data X"""
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
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.transform_with_z(X,zfile,zids)
        else:
            return self._pyfeat.transform(X)

    def fit_predict(self,X,y):
        """Convenience method that runs fit(X,y) then predict(X)"""
        self._init_pyfeat()    
        return self._pyfeat.fit_predict(X,y)

    def fit_transform(self,X,y):
        """Convenience method that runs fit(X,y) then transform(X)"""
        self._init_pyfeat()    
        return self._pyfeat.fit_transform(X,y)

    def score(self,X,y,zfile=None,zids=None):
        """Returns a score for the predictions of Feat on X versus true labels y""" 
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
        """Returns a string with the set of equations and weights in the final representation"""
        return self._pyfeat.get_model()

    def get_representation(self):
        """Returns a string with the final representation"""
        return self._pyfeat.get_representation()

    def get_archive(self):
        """Returns all the final representation equations in the archive"""
        return self._pyfeat.get_archive()

    def get_coefs(self):
        """Returns the coefficients assocated with each feature in the representation"""
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
        
    def get_gens(self):
        """return generations statistics arrays"""
        return self._pyfeat.get_gens()
              
    def get_timers(self):
        """return time statistics arrays"""
        return self._pyfeat.get_timers()
        
    def get_best_scores(self):
        """return best score statistics arrays"""
        return self._pyfeat.get_best_scores()
        
    def get_best_score_vals(self):
        """return best score values statistics arrays"""
        return self._pyfeat.get_best_score_vals()
        
    def get_med_scores(self):
        """return median scores statistics arrays"""
        return self._pyfeat.get_med_scores()
        
    def get_med_loss_vals(self):
        """return median loss values statistics arrays"""
        return self._pyfeat.get_med_loss_vals()
        
    def get_med_size(self):
        """return median size statistics arrays"""
        return self._pyfeat.get_med_size()
        
    def get_med_complexities(self):
        """return median complexity statistics arrays"""
        return self._pyfeat.get_med_complexities()
        
    def get_med_num_params(self):
        """return median num params statistics arrays"""
        return self._pyfeat.get_med_num_params()
        
    def get_med_dim(self):
        """return median dimensions statistics arrays"""
        return self._pyfeat.get_med_dim()
