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

        self._pyfeat = pyfeat.PyFeat( pop_size,  gens,  ml, 
                classification,  verbosity,  max_stall,
                sel,  surv,  cross_rate,
                otype,  functions, 
                max_depth,   max_dim,  random_state, 
                erc,  obj, shuffle,  split,  fb)

    def fit(self,X,y):
        self._pyfeat.fit(X,y)

    def predict(self,X):
        self._pyfeat.predict(X)

    def transform(self,X):
        self._pyfeat.transform(X)

    def fit_predict(self,X,y):
        self._pyfeat.fit_predict(X,y)

    def fit_transform(self,X,y):
        self._pyfeat.fit_transform(X,y)


def main():
    """Main function that is called when Fewtwo is run from the command line"""
    parser = argparse.ArgumentParser(description="A feature engineering wrapper for ML.",
                                     add_help=False)

    clf = Feat()
if __name__ == '__main__':
    main()
