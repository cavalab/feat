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
from sklearn.metrics import accuracy_score,mean_squared_error as mse


class Feat(BaseEstimator):
    """Feat uses GP to find a data representation that improves the performance of a given ML
    method."""
    def __init__(self, pop_size=100,  gens=100,  ml="LinearRidgeRegression", 
                classification=False,  verbosity=0,  max_stall=0,
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

    def getAverageMethodForScore(labels):
        labels_test_set = set(labels)
        average_method = 'binary'
        if len(labels_test_set) >2:
            average_method = 'macro'
        return average_method

    def score(self,features,labels):
        labels_pred = self.predict(features)
        if ( self.classification ):
            return accuracy_score(labels,labels_pred,average=getAverageMethodForScore(labels))
        else:
            return mse(labels,labels_pred)
        

def main():
    """Main function that is called when Fewtwo is run from the command line"""
    parser = argparse.ArgumentParser(description="A feature engineering wrapper for ML.",
                                     add_help=False)
    parser.add_argument('-p', action='store', dest='POPULATION',default=100,type=int, help='Population Size')
    parser.add_argument('-g', action='store', dest='GENERATIONS',default=100,type=int, help='Generations Size')    
    parser.add_argument('-ml', action='store', dest='ML',
                        default='LinearRidgeRegression',
                        choices = ['LinearRidgeRegression','LogisticRegression'],type=str, help='Machine Learning Modelling Pair')      
    parser.add_argument('-c', action='store', dest='IS_CLASSIFICATION',default=False,type=bool, help='False if Classification (default)')
    parser.add_argument('-v', action='store', dest='VERBOSITY',default=0,type=int, help='Debugging Level 0 | 1 | 2 ')
    parser.add_argument('-stall', action='store', dest='MAX_STALL',default=0,type=int, help='Maximum generations with no improvement to best score')
    parser.add_argument('-sel', action='store', dest='SELECTION_METHOD',
                        default='lexicase',
                        type=str, help='Selection Method')  
    parser.add_argument('-surv', action='store', dest='SURVIVAL_METHOD',
                        default='pareto',
                        type=str, help='Survival Method')
    parser.add_argument('-xr', action='store', dest='CROSS_OVER',default=0.5,type=float, help='Cross over Rate in [0,1]')
    parser.add_argument('-otype', action='store', dest='O_TYPE',default="a",type=str, help='OType')
    parser.add_argument('-f', action='store', dest='FUNCTIONS',default="+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite",type=str, help='Terminal Functions')
    parser.add_argument('-mdep', action='store', dest='MAX_DEPTH',default=3,type=int, help='Max Depth')
    parser.add_argument('-mdim', action='store', dest='MAX_DIMENSION',default=10,type=int, help='Max Dimension')
    parser.add_argument('-rs', action='store', dest='RANDOM_STATE',default=0,type=int, help='Random State')
    parser.add_argument('-erc', action='store', dest='ERC',default=False,type=bool, help='ERC')
    parser.add_argument('-split', action='store', dest='SPLIT',default=0.75,type=float, help='Split ratio for feat')
    parser.add_argument('-fb', action='store', dest='FB',default=0.5,type=float, help='Fb')
    parser.add_argument('-shuffle', action='store', dest='SHUFFLE',default=False,type=bool, help='False if no shuffle')
    parser.add_argument('-obj', action='store', dest='OBJ',default="fitness,complexity",type=str, help='Obj')

    args = parser.parse_args()

    learner = Feat(pop_size = args.POPULATION,
         gens = args.GENERATIONS,
         ml = args.ML,
         classification = args.IS_CLASSIFICATION,
         verbosity = args.VERBOSITY,
         max_stall = args.MAX_STALL,
         sel = args.SELECTION_METHOD,
         surv = args.SURVIVAL_METHOD,
         cross_rate = args.CROSS_OVER,
         otype = args.O_TYPE,
         functions = args.FUNCTIONS,
         max_depth = args.MAX_DEPTH,
         max_dim = args.MAX_DIMENSION,
         random_state = args.RANDOM_STATE,
         erc = args.ERC,
         split = args.SPLIT,
         fb = args.SHUFFLE,
         obj = args.OBJ
        )

if __name__ == '__main__':
    main()
