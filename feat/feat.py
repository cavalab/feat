# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""

import argparse
from .versionstr import __version__
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd
from _feat import cppFeat
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss
from sklearn.utils import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder
import json

class Feat(BaseEstimator):
    """Feature Engineering Automation Tool

    Parameters
    ----------
    pop_size: int, optional (default: 100)
        Size of the population of models
    gens: int, optional (default: 100)
        Number of iterations to train for
    ml: str, optional (default: "LinearRidgeRegression")
        ML pairing. Choices: LinearRidgeRegression, Lasso, L1\_LR, L2\_LR
        FeatRegressor sets to "LinearRidgeRegression";
        FeatClassifier sets to L2 penalized LR ("LR")
    classification: boolean or None, optional (default: None)
        Whether to do classification or regression. Set explicitly in 
        FeatRegressor and FeatClassifier accordingly.
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
    functions: list[string], optional (default: [])
        A comma-separated string of operators to use to build features. 
        If functions=[], all the available functions are used. 
        Options: +, -, *, /, ^2, ^3, sqrt, sin, cos, exp, log, ^, logit, tanh, 
        gauss, relu, split, split_c, b2f, c2f, and, or, not, xor, =, <, <=, >,
        >=, if, ite
    max_depth: int, optional (default: 3)
        Maximum depth of a feature's tree representation.
    max_dim: int, optional (default: 10)
        Maximum dimension of a model. The dimension of a model is how many
        independent features it has. Controls the number of trees in each 
        individual.
    random_state: int, optional (default: 0)
        Random seed. If -1, will choose a random random_state.
    erc: boolean, optional (default: False)
        If true, ephemeral random constants are included as nodes in trees.
    objectives: list[str], optional (default: "fitness,complexity")
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
        Saves the population of models. 0: don't save; 1: save final 
        population; 2: save every generation. 
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

    __version__ = __version__

    def __init__(self, 
                 pop_size=100, 
                 gens=100, 
                 ml= 'LinearRidgeRegression', 
                 classification=False,
                 verbosity=0, 
                 max_stall=0, 
                 sel="lexicase", 
                 surv="nsga2", 
                 cross_rate=0.5, 
                 root_xo_rate=0.5, 
                 otype='a', 
                 functions= [
                    "+","-","*","/","^2","^3","sqrt","sin","cos","exp","log","^",
                    "logit","tanh","gauss","relu", "split","split_c",
                    "b2f","c2f","and","or","not","xor","=","<","<=",">",">=","if","ite"
                 ],
                 max_depth=3, 
                 max_dim=10, 
                 random_state=0, 
                 erc= False , 
                 objectives=["fitness","complexity"], 
                 shuffle=True, 
                 split=0.75, 
                 fb=0.5, 
                 scorer='', 
                 feature_names="", 
                 backprop=False, 
                 iters=10, 
                 lr=0.1, 
                 batch_size=0, 
                 n_jobs=1, 
                 hillclimb=False, 
                 logfile="", 
                 max_time=-1, 
                 residual_xo=False, 
                 stagewise_xo=False, 
                 stagewise_xo_tol=False, 
                 softmax_norm=False, 
                 save_pop=0, 
                 normalize=True, 
                 val_from_arch=False, 
                 corr_delete_mutate=False, 
                 simplify=0.0, 
                 protected_groups="", 
                 tune_initial=False, 
                 tune_final=True, 
                 starting_pop="",
                ):
        self.pop_size=pop_size
        self.gens=gens
        self.ml=ml
        self.classification = classification
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
        self.objectives=objectives
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
        
    def load(self, filename):
        """Load a saved Feat state from file."""
        with open(filename, 'r') as of:
            feat_state = json.load(of)
        # separate out the python-specific keys
        init_params = {k:v for k,v in feat_state.items() if not k.endswith('_')}
        self.set_params(**init_params) 
        for k,v in feat_state.items():
            if k.endswith('_') and k != 'cfeat_':
                setattr(self, k, v)

        if 'cfeat_' in feat_state.keys():
            self.cfeat_ = cppFeat()
            self.cfeat_.load(feat_state['cfeat_'])
        else:
            self._set_cfeat_params()

        return self

    def save(self, filename):
        """Save a Feat state to file."""

        cfeat_state = self.cfeat_.save()
        # add python-specific parameters
        attribs = self.get_params()
        for k,v in self.__dict__.items(): 
            if k not in attribs.keys():
                attribs[k] = v
        attribs['cfeat_']  = cfeat_state
        with open(filename, 'w') as of:
            json.dump(attribs, of)

    def _set_cfeat_params(self):
        """Setup cpp feat estimator."""
        self.cfeat_ = cppFeat()
        params = self.get_params()
        for k,v in params.items():
            setattr(self.cfeat_, k, v)

    def fit(self, X, y, Z=None):
        """Fit a model."""    

        X,y = self._clean(X, y, set_feature_names=True)

        self._set_cfeat_params() 

        if Z:
            self.cfeat_.fit(X,y,Z)
        else:
            self.cfeat_.fit(X,y)

        self.is_fitted_ = True
        return self

    def predict(self,X,Z=None):
        """Predict on X."""
        if not self.is_fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._prep_X(X)

        if Z:
            return self.cfeat_.predict(X,Z)
        else:
            return self.cfeat_.predict(X)

    def predict_archive(self,X,Z=None,front=False):
        """Returns a list of dictionary predictions for all models."""
        if not self.is_fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._prep_X(X)

        if Z:
            raise NotImplementedError('longitudinal not implemented')
            return

        archive = self.cfeat_.get_archive(front)
        preds = []
        for ind in archive:
            # if ind['id'] == 9234:
            #     print('individual:',json.dumps(ind,indent=2))
            tmp = {}
            tmp['id'] = ind['id']
            tmp['y_pred'] = self.cfeat_.predict_archive(ind['id'], X) 
            preds.append(tmp)

        return preds

    def transform(self,X,Z=None):
        """Return the representation's transformation of X"""
        if not self.is_fitted_:
            raise ValueError("Call fit before calling transform.")

        X = self._prep_X(X)

        if Z:
            Phi =  self.cfeat_.transform(X,Z)
        else:
            Phi = self.cfeat_.transform(X)

        return Phi

    def fit_predict(self,X,y,Z=None):
        """Convenience method that runs fit(X,y) then predict(X)"""
        self.fit(X,y,Z)
        result = self.predict(X,Z)
        return result

    def fit_transform(self,X,y,Z=None):
        """Convenience method that runs fit(X,y) then transform(X)"""
        return self.fit(X,y,Z).transform(X,Z)

    def score(self,X,y,Z=None):
        """Returns a score for the predictions of Feat on X versus true 
        labels y""" 
        if ( self.classification ):
            yhat = self.predict_proba(X,Z)
            return log_loss(y, yhat, labels=y)
        else:
            yhat = self.predict(X,Z).flatten()
            return mse(y,yhat)


    @property
    def stats_(self): 
        if not self.is_fitted_:
            raise ValueError("Call fit before asking for stats_.")
        return self.cfeat_.stats_

    def _prep_array(self, x):
        """Converts dataframe to array, optionally returning feature names"""
        x = np.asfortranarray(x, dtype=np.float32)
        return x

    def _prep_X(self, X):
        X = check_array(X)
        self._check_shape(X)
        return self._prep_array(X).transpose()

    def _clean(self, x, y, set_feature_names=False):
        """Converts dataframe to array, optionally returning feature names"""
        feature_names = ''
        if isinstance(x, pd.DataFrame):
            if set_feature_names and len(list(x.columns)) == x.shape[1]:
                self.feature_names = ','.join(x.columns)
        x, y = check_X_y(x,y, ensure_min_samples=2, dtype=np.float32)
        self.n_features_in_ = x.shape[1]
        x = self._prep_X(x)
        y = self._prep_array(y)

        return x,y

    def _check_shape(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError('The number of features ({}) in X do not match '
                             'the number of features used for training '
                             '({})'.format(X.shape[1],self.n_features_in_))
    # wrappers
    def get_representation(self): return self.cfeat_.get_representation()
    def get_model(self, sort=True): return self.cfeat_.get_model(sort)
    def get_coefs(self): return self.cfeat_.get_coefs()
    def get_n_params(self): return self.cfeat_.get_n_params()
    def get_complexity(self): return self.cfeat_.get_complexity()
    def get_dim(self): return self.cfeat_.get_dim()
    def get_n_nodes(self): return self.cfeat_.get_n_nodes()

class FeatRegressor(Feat):
    """Convenience method that enforces regression options."""
    def __init__(self,**kwargs):
        kwargs.update({'classification':False})
        if 'ml' not in kwargs: 
            kwargs['ml'] = 'LinearRidgeRegression'
        Feat.__init__(self,**kwargs)
    def _get_param_names(self):
        return Feat._get_param_names()

class FeatClassifier(Feat):
    """Convenience method that enforces classification options.
        Also includes methods for prediction probabilities.
    """
    def __init__(self,**kwargs):
        kwargs.update({'classification':True})
        if 'ml' not in kwargs: 
            kwargs['ml'] = 'LR'
        Feat.__init__(self,**kwargs)

    def _get_param_names(self):
        return Feat._get_param_names()
    def fit(self,X,y,zfile=None,zids=None):
        self.classes_ = [int(i) for i in np.unique(np.asarray(y))]
        if (any([
            i != j 
            for i,j in zip(self.classes_, np.arange(np.max(self.classes_)))
        ])):
            raise ValueError('y must be a contiguous set of labels from ',
                             '0 to n_classes. y contains the values {}'.format(
                                self.classes_)
                            )
       
        super().fit(X,y)
        return self

    def predict(self,X,Z=None):
        return super().predict(X, Z)

    def predict_proba(self,X,Z=None):
        """Return probabilities of predictions for data X"""
        if not self.is_fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._prep_X(X)

        if Z:
            tmp = self.cfeat_.predict_proba(X,Z).transpose()
        else:
            tmp = self.cfeat_.predict_proba(X).transpose()
        
        # for binary classification, add a second column for 0 complement
        if len(self.classes_) == 2 and tmp.shape[1] == 1:
            tmp = tmp.ravel()
            assert X.shape[1] == len(tmp)
            tmp = np.vstack((1-tmp,tmp)).transpose()
        return tmp         

    def predict_proba_archive(self,X,Z=None,front=False):
        """Returns a dictionary of prediction probabilities for all models."""

        if not self.is_fitted_:
            raise ValueError("Call fit before calling predict.")

        X = self._prep_X(X)

        archive = self.cfeat_.get_archive(front)
        probs = []
        for ind in archive:
            tmp = {}
            tmp['id'] = ind['id']
            if Z:
                tmp['y_proba'] = self.cfeat_.predict_proba_archive(ind['id'], X, Z)
            else:
                tmp['y_proba'] = self.cfeat_.predict_proba_archive(ind['id'], X)
            probs.append(tmp)

        return probs
