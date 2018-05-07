# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava
license: GNU/GPLv3
"""
from feat import Feat
import numpy
from sklearn.datasets import load_diabetes
import unittest
import argparse
import sys

verbosity = 0

class TestFeatWrapper(unittest.TestCase):

    def setUp(self):
        self.v = verbosity
        self.clf = Feat(verbosity=self.v)
        diabetes = load_diabetes()
        self.X = diabetes.data
        self.y = diabetes.target
        
    #Test 1: Assert the length of labels returned from predict
    def test_predict_length(self):
        self.debug("Fit the Data")
        self.clf.fit(self.X,self.y)

        self.debug("Predicting the Results")
        pred = self.clf.predict(self.X)

        self.debug("Comparing the Length of labls in Predicted vs Actual ")
        expected_length = len(self.y)
        actual_length = len(pred)
        self.assertEqual( actual_length , expected_length )

    #Test 2:  Assert the length of labels returned from fit_predict
    def test_fitpredict_length(self):
        self.debug("Calling fit_predict from Feat")
        pred = self.clf.fit_predict(self.X,self.y)

        self.debug("Comparing the length of labls in fit_predict vs actual ")
        expected_length = len(self.y)
        actual_length = len(pred)
        self.assertEqual( actual_length , expected_length )

    #Test 3:  Assert the length of labels returned from transform
    def test_transform_length(self):
        self.debug("Calling fit")
        self.clf.fit(self.X,self.y)
        trans_X = self.clf.transform(self.X)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )

    #Test 4:  Assert the length of labels returned from fit_transform
    def test_fit_transform_length(self):
        self.debug("In wrappertest.py...Calling fit transform")
        trans_X = self.clf.fit_transform(self.X,self.y)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )
        
    #Test 5:  Transform with Z
    def test_transform_length_z(self,zfile=None,zids=None):
        self.debug("Calling fit")
        self.clf.fit(self.X,self.y)
        trans_X = self.clf.transform(self.X,zfile,zids)

        self.debug("Comparing the length of labls in transform vs actual feature set ")
        expected_value = self.X.shape[0]
        actual_value = trans_X.shape[0]
        self.assertEqual( actual_value , expected_value )

    def debug(self,message):
        if ( self.v > 0 ):
            print (message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="",add_help=False)
   
    parser.add_argument('-v', action='store', dest='VERBOSE',default=0,type=int, help='0 for no Verbose. 1 for Verbosity')

    args = parser.parse_args()
    verbosity = args.VERBOSE
    if len(sys.argv) > 1:
        sys.argv.pop()
        
    unittest.main()
        
