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

class Feat(BaseEstimator):
    """Feat uses GP to find a data representation that improves the performance of a given ML
    method."""
    def __init__(self, pop_size=100,  gens=100,  ml="LinearRidgeRegression", 
                classification=False,  verbosity=2,  max_stall=0,
                sel="lexicase",  surv="pareto",  cross_rate=0.5,
                otype='a',  functions="+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite", 
                max_depth=3,   max_dim=10,  random_state=0, 
                erc = False,  obj="fitness,complexity", shuffle=False,  split=0.75,  fb=0.5):
        self.pop_size = pop_size
        self.gens = gens
        self.ml = ml
        self.classification = classification
        self.verbosity = verbosity
        self.max_stall = max_stall
        self.sel = sel
        self.surv = surv
        self.cross_rate = cross_rate
        self.otype = otype
        self.functions = functions
        self.max_depth = max_depth
        self.max_dim = max_dim
        self.random_state = random_state
        self.erc = erc      
        self.obj = obj
        self.shuffle = shuffle
        self.split = split
        self.fb = fb
   
        self._pyfeat = pyfeat.PyFeat( self.pop_size,  self.gens,  self.ml, 
                self.classification,  self.verbosity,  self.max_stall,
                self.sel,  self.surv,  self.cross_rate,
                self.otype,  self.functions, 
                self.max_depth,   self.max_dim,  self.random_state, 
                self.erc,  self.obj, self.shuffle,  self.split,  self.fb)

    def fit(self,X,y):
        self._pyfeat.fit(X,y)

    def predict(self,X):
        return self._pyfeat.predict(X)

    def transform(self,X):
        return self._pyfeat.transform(X)

    def fit_predict(self,X,y):
        return self._pyfeat.fit_predict(X,y)

    def fit_transform(self,X,y):
        return self._pyfeat.fit_transform(X,y)


def main():
    """Main function that is called when Fewtwo is run from the command line"""
    parser = argparse.ArgumentParser(description="A feature engineering wrapper for ML.",
                                     add_help=False)

    
if __name__ == '__main__':
    main()
