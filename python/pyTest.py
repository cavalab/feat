import feat
import numpy
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data


Y = boston.target

clf = feat.PyFewtwo()
clf.fit(X,Y)
#print  clf.predict(Y) 
