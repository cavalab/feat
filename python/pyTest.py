from feat import *
import pyfeat
import numpy
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data


Y = boston.target

clf = pyfeat.pyfeat(split=0.5,verbosity=0)
clf.fit(X,Y)


print clf.predict(X)
print ( clf.transform(X))
print clf.fit_predict(X,Y)
print clf.fit_transform(X,Y)

