import feat
import numpy
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data


Y = boston.target

clf = feat.PyFewtwo(split=0.5,verbosity=0)
clf.fit(X,Y)
print clf.predict(X)
#print  clf.predict(Y) 
