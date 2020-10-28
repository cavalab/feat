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

    def __init__(self, **kwargs):
        return PyFeat.__init__(self, **kwargs)
    # def get_params(self, deep=False):
    #     if deep:
    #         return self.get_params()
    #     return self.params
   
    def get_params(self, deep=False):
        # for name in dir(self.__class__):
        #     print('getting',name)
        #     obj = getattr(self.__class__, name)
        #     if isinstance(obj, property):
        #         val = obj.__get__(self, self.__class__)
        params = {k[1:] : v for k, v in self.__dict__.items() 
                if k.startswith("_") and not k.endswith("_")}
        print('returning params:',params)
        return params

    def fit(self,X,y,zfile=None,zids=None):
        """Fit a model."""    
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
