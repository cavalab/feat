# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

import argparse
from ._version import __version__

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from fewtwo import PyFewtwo

class Feat(BaseEstimator):
    """Feat uses GP to find a data representation that improves the performance of a given ML
    method."""
    def __init__(self):

    def fit(self,X,y):

    def predict(self,X):

    def transform(self,X):

    def fit_predict(self,X,y):

    def score(self,X,y):


def main():
    """Main function that is called when Fewtwo is run from the command line"""
    parser = argparse.ArgumentParser(descripion="A feature engineering wrapper for ML.",
                                     add_help=False)

if __name__ == '__main__':
    main()
