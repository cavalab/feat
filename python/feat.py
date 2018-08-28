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

class Feat(BaseEstimator):
    """Feat uses GP to find a data representation that improves the performance of a given ML
    method."""
    def __init__(self, pop_size=100,  gens=100,  ml = "LinearRidgeRegression", 
                classification=False,  verbosity=0,  max_stall=0,
                sel ="lexicase",  surv ="nsga2",  cross_rate=0.5,
                otype ='a',  functions ="", 
                max_depth=3,   max_dim=10,  random_state=0, 
                erc = False,  obj ="fitness,complexity", shuffle=False,  split=0.75,  fb=0.5,
                scorer ='',feature_names="", backprop=False, iters=10, lr=0.1, batch_size=100, n_threads=0,
                hillclimb=False, logfile="Feat.log", max_time=-1, use_batch=False):
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
        self.feature_names = feature_names.encode() if isinstance(feature_names,str) else feature_names 
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
        # if self.verbosity>0:
        #print('self.__dict__: ' , self.__dict__)
        self._pyfeat=None 

    def _init_pyfeat(self):
        # set up pyfeat glue class object
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
                self.use_batch)
   
    def fit(self,X,y,zfile=None,zids=None):
        
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

    def predict(self,X,zfile=None,zids=None):
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.predict_with_z(X,zfile,zids)
        else:
            return self._pyfeat.predict(X)

    def predict_proba(self,X,zfile=None,zids=None):
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            tmp = self._pyfeat.predict_proba_with_z(X,zfile,zids)
        else:
            tmp = self._pyfeat.predict_proba(X)
        
        if len(tmp.shape)<2:
                tmp  = np.vstack((1-tmp,tmp)).transpose()
        return tmp         


    def transform(self,X,zfile=None,zids=None):
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            return self._pyfeat.transform_with_z(X,zfile,zids)
        else:
            return self._pyfeat.transform(X)

    def fit_predict(self,X,y):
        self._init_pyfeat()    
        return self._pyfeat.fit_predict(X,y)

    def fit_transform(self,X,y):
        self._init_pyfeat()    
        return self._pyfeat.fit_transform(X,y)

    def score(self,features,labels,zfile=None,zids=None):
        if zfile:
            zfile = zfile.encode() if isinstance(zfile,str) else zfile
            labels_pred = self._pyfeat.predict_with_z(features,zfile,zids).flatten()
        else:
            labels_pred = self.predict(features).flatten()
        if ( self.classification ):
            return log_loss(labels,labels_pred, labels=labels)
        else:
            return mse(labels,labels_pred)

    def get_model(self):
        return self._pyfeat.get_model()

    def get_representation(self):
        return self._pyfeat.get_representation()

    def get_archive(self):
        return self._pyfeat.get_archive()

    def get_coefs(self):
        return self._pyfeat.get_coefs()

    def get_dim(self):
        return self._pyfeat.get_dim()

    def get_n_params(self):
        return self._pyfeat.get_n_params()

    def get_complexity(self):
        return self._pyfeat.get_complexity()

def main():
    """Main function that is called when Fewtwo is run from the command line"""
    parser = argparse.ArgumentParser(description="A feature engineering wrapper for ML.",
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to run FEW on; ensure that the '
                        'target/label column is labeled as "label" or "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-p', action='store', dest='POPULATION',default=100,type=int, help='Population Size')
    parser.add_argument('-g', action='store', dest='GENERATIONS',default=100,type=int, help='Generations Size')    
    parser.add_argument('-ml', action='store', dest='ML',
                        default='LinearRidgeRegression',
                        choices = ['LinearRidgeRegression','LogisticRegression'],type=str, help='Machine Learning Modelling Pair')      
    parser.add_argument('--c', action='store_true', dest='IS_CLASSIFICATION',default=False, help='Do classification')
    parser.add_argument('-v', action='store', dest='VERBOSITY',default=0,type=int, help='Debugging Level 0 | 1 | 2 ')
    parser.add_argument('-stall', action='store', dest='MAX_STALL',default=0,type=int, help='Maximum generations with no improvement to best score')
    parser.add_argument('-sel', action='store', dest='SELECTION_METHOD',
                        default='lexicase',
                        type=str, help='Selection Method')  
    parser.add_argument('-sep', action='store', dest='SEP', default=',', type=str, help='separator used on input file.')
    parser.add_argument('-surv', action='store', dest='SURVIVAL_METHOD',
                        default='nsga2',
                        type=str, help='Survival Method')
    parser.add_argument('-xr', action='store', dest='CROSS_OVER',default=0.5,type=float, help='Cross over Rate in [0,1]')
    parser.add_argument('-otype', action='store', dest='O_TYPE',default="a",type=str, help='OType')
    parser.add_argument('-ops', action='store', dest='FUNCTIONS',default="+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite",type=str, help='Terminal Functions')
    parser.add_argument('-depth', action='store', dest='MAX_DEPTH',default=3,type=int, help='Max Depth')
    parser.add_argument('-dim', action='store', dest='MAX_DIMENSION',default=10,type=int, help='Max Dimension')
    parser.add_argument('-r', action='store', dest='RANDOM_STATE',default=0,type=int, help='Random State')
    parser.add_argument('-erc', action='store', dest='ERC',default=False, help='Include ephemeral random constants')
    parser.add_argument('-split', action='store', dest='SPLIT',default=0.75,type=float, help='Split ratio for feat')
    parser.add_argument('-fb', action='store', dest='FB',default=0.5,type=float, help='Fb')
    parser.add_argument('--shuffle', action='store_true', dest='SHUFFLE',default=False, help='False if no shuffle')
    parser.add_argument('-obj', action='store', dest='OBJ',default="fitness,complexity",type=str, help='Obj')

    args = parser.parse_args() 
    
    #
    # set up the input data
    #

    # load data from csv file
    input_data = pd.read_csv(args.INPUT_FILE, sep=args.SEP,
                                 engine='python')

    # if 'Label' in input_data.columns.values:
    input_data.rename(columns={'Label': 'label','Class':'label','class':'label',
                               'target':'label'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE

    train_i, test_i = train_test_split(input_data.index,
                                       stratify = None,
                                       #stratify=input_data['label'].values,
                                       train_size=0.75,
                                       test_size=0.25,
                                       random_state=RANDOM_STATE)

    training_features = input_data.loc[train_i].drop('label', axis=1).values
    training_labels = input_data.loc[train_i, 'label'].values
    
    testing_features = input_data.loc[test_i].drop('label', axis=1).values
    testing_labels = input_data.loc[test_i, 'label'].values

  
    #
    # set up a Feat learner
    #
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
    # fit model
    learner.fit(training_features, training_labels)
    # print results    
    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {:1.3f}'.format(
            learner.score(training_features, training_labels)))
        print('Test accuracy: {:1.3f}'.format(
            learner.score(testing_features, testing_labels)))

if __name__ == '__main__':
    main()
