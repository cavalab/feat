import feat
import numpy
from sklearn.datasets import load_breast_cancer

data_iris = load_breast_cancer()
X = data_iris.data


Y = data_iris.target

clf = feat.PyFewtwo()
clf.fit(X,Y)
#print  clf.predict(Y) 
