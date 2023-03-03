# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""
from feat import Feat, FeatRegressor, FeatClassifier 

from sklearn.datasets import load_diabetes, make_blobs
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import rbf_kernel
import unittest
import argparse
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.utils.estimator_checks import (
    check_estimator,
    _enforce_estimator_tags_y
)
from sklearn.utils._testing import (
    set_random_state,
    assert_allclose_dense_sparse
)

verbosity = 2


class TestFeatWrapper(unittest.TestCase):

    def setUp(self):
        """setup"""
        self.v = verbosity
        self.reg = FeatRegressor(verbosity=verbosity, 
                                 n_jobs=1, 
                                 gens=2,
                                 random_state=42
                                )
        self.clf = FeatClassifier(verbosity=verbosity, 
                                  n_jobs=1, 
                                  gens=2,
                                  random_state=42,
                                 )
        diabetes = load_diabetes()
        self.X = diabetes.data
        self.yr = diabetes.target
        self.yc = diabetes.target < np.median(diabetes.target)
        self.yc = np.array([int(i) for i in self.yc])

    def debug(self,message):
        if ( self.v > 0 ):
            print (message)

    def test_sklearn_api(self):
        """tests sklearn api"""
        # clf = self.clf
        # clf.classification = True
        self.debug("Check Estimator")
        check_generator = check_estimator(self.clf, generate_only=True)
        check_generator2 = check_estimator(self.reg, generate_only=True)

        
        #TODO: make these checks pass.
        skip_checks = ['check_estimators_pickle',
                       'check_methods_subset_invariance',
                       'check_fit2d_1sample',
                       'check_fit2d_1feature',
                       'check_transformer_data_not_an_array',
                       'check_transformer_preserve_dtypes' 
                      ]
        for est, check in check_generator2:
            time_to_go=False
            for ch in skip_checks:
                if ch in str(check):
                    time_to_go = True
            if time_to_go: continue
            check(est)

        for est, check in check_generator:
            time_to_go=False
            for ch in skip_checks:
                if ch in str(check):
                    time_to_go = True
                    break
            if time_to_go: continue
            check(est)

    #Test 1: Assert the length of labels returned from predict
    def test_predict_length(self):
        """prediction length"""
        self.debug("Fit the Data")
        self.clf.fit(self.X,self.yc)

        self.debug("Predicting the Results")
        pred = self.clf.predict(self.X)

        self.debug("Comparing the Length of labls in Predicted vs Actual ")
        expected_length = len(self.yr)
        actual_length = len(pred)
        self.assertEqual( actual_length , expected_length )

    #Test 2:  Assert the length of labels returned from fit_predict
    def test_fitpredict_length(self):
        """fitpredict length"""
        self.debug("Calling fit_predict from Feat")
        
        pred = self.reg.fit_predict(self.X,self.yr)

        self.debug("Comparing the length of labls in fit_predict vs actual ")
        expected_length = len(self.yr)
        actual_length = len(pred)
        self.assertEqual( actual_length , expected_length )

    #Test 3:  Assert the length of labels returned from transform
    def test_transform_length(self):
        self.debug("Calling fit")
        self.reg.fit(self.X,self.yr)
        trans_X = self.reg.transform(self.X)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )

    #Test 4:  Assert the length of labels returned from fit_transform
    def test_fit_transform_length(self):
        self.debug("In wrappertest.py...Calling fit transform")
        trans_X = self.reg.fit_transform(self.X,self.yr)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )
        
    #Test 5:  Transform with Z
    def test_transform_length_z(self,Z=None):
        self.debug("Calling fit")
        self.reg.fit(self.X,self.yr)
        trans_X = self.reg.transform(self.X,Z)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )


    def test_coefs(self):
        self.debug("In wrappertest.py...Calling test_coefs")
        self.reg.fit(self.X,self.yr)
        coefs = self.reg.get_coefs()
        self.assertTrue( len(coefs)>0 )

    def test_dataframe(self):
        self.debug("In wrappertest.py...Calling test_dataframe")
        dfX = pd.DataFrame(data=self.X,columns=['fishy'+str(i) 
                                        for i in np.arange(self.X.shape[1])],
                                        index=None)
        dfy = pd.DataFrame(data={'label':self.yr})

        self.reg.fit(dfX,dfy['label'])
        assert(self.reg.feature_names == ','.join(dfX.columns))

    #Test: Assert the length of labels returned from predict
    def test_predict_stats_length(self):
        self.debug("Fit the Data")
        self.reg.fit(self.X,self.yr)

        for key,value in self.reg.stats_.items():
            self.assertEqual(len(value), self.reg.gens)

    #Test ability to pickle feat model
    def test_save_load(self):
        self.debug("Test saving and reloading Feat objects")
        # old_verbose = self.reg.verbosity
        # self.reg.verbosity = 3 
        reg = clone(self.reg) 
        reg.fit(self.X, self.yr)
        initial_pred = reg.predict(self.X)
        reg.save('Feat_tmp.json')

        loaded_reg = FeatRegressor().load('Feat_tmp.json')
        loaded_pred = loaded_reg.predict(self.X)
        diff = np.abs(initial_pred-loaded_pred)
        cov = np.corrcoef(initial_pred, loaded_pred)[1,1]
        # print('initial pred:',initial_pred)
        # print('loaded pred:',loaded_pred)
        # print('diff',diff)
        if cov == 1 and np.mean(diff)>0.0001:
            print('loaded regressor has the wrong scale/offset.',
                  'probably a normalization issue.')


        for i,d in enumerate(diff):
            if d > 0.0001:
                print('pred:',initial_pred[i],'loaded:',loaded_pred[i],
                      'offset:',d,'scale:',initial_pred[i]/loaded_pred[i])
        assert all([d < 0.0001 for d in diff])

        assert(reg.get_representation() == loaded_reg.get_representation())
        assert(reg.get_model() == loaded_reg.get_model())
        assert((reg.get_coefs() == loaded_reg.get_coefs()).all())
        loaded_params = loaded_reg.get_params()
        # print('\n',10*'=','\n')
        # print('loaded_params:')
        # for k,v in loaded_params.items():
        #     print(k,':',v)

        for k,v in reg.get_params().items():
            if k not in loaded_params.keys():
                print(k,'not in ',loaded_params.keys())
                assert(k in loaded_params.keys())
            if isinstance(v,float):
                if np.abs(loaded_params[k] - v) > 0.0001:
                    print('loaded_params[',k,'] =',
                      loaded_params[k], '\nwhich is different from:', v)
                assert(np.abs(loaded_params[k] - v) < 0.0001)
            elif loaded_params[k] != v:
                print('loaded_params[',k,'] =',
                      loaded_params[k], '\nwhich is different from:', v)
                assert(loaded_params[k] == v)

        loaded_reg.fit(self.X, self.yr)

    def test_archive(self):
        """test archiving ability"""
        self.debug("Test archive")
        self.clf.fit(self.X,self.yc)
        self.debug('grabbing archive..')
        archive = self.clf.cfeat_.get_archive()
        self.debug('grabbing predictions..')
        preds = self.clf.predict_archive(self.X)
        self.debug('grabbing prediction probs..')
        probs = self.clf.predict_proba_archive(self.X)

        for arch, pred, prob in zip(archive, preds, probs):
            self.assertTrue( arch['id'] == pred['id'] )
            self.assertTrue( arch['id'] == prob['id'] )

    def test_lr_l1(self):
        """testing l1 penalized LR"""
        self.clf.ml = 'L1_LR'
        self.clf.fit(self.X,self.yc)

        self.assertEqual(len(self.clf.predict(self.X)), len(self.yc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="",add_help=False)
   
    parser.add_argument('-v', action='store', dest='VERBOSE',default=0,type=int, 
            help='0 for no Verbose. 1 for Verbosity')

    args = parser.parse_args()
    verbosity = args.VERBOSE
    if len(sys.argv) > 1:
        sys.argv.pop()
        
    unittest.main()
        
