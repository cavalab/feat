import feat1
import pyfeat
import numpy
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data


Y = boston.target

#clf= feat1.Feat()
clf = pyfeat.PyFeat(pop_size=100,  gens=100,  ml="LinearRidgeRegression", classification=False,  verbosity=0, max_stall=0,sel="lexicase",  surv="pareto",cross_rate=0.5,otype="a",functions="+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite", max_depth=3,   max_dim=10,  random_state=0, erc = False,  obj="fitness,complexity", shuffle=False,  split=0.75,  fb=0.5)

clf.fit(X,Y)


print clf.predict(X)
print ( clf.transform(X))
print clf.fit_predict(X,Y)
print clf.fit_transform(X,Y)

# pop_size=100,  gens=100,  ml="LinearRidgeRegression", classification=False,  verbosity=2,  max_stall=0,sel="lexicase",  surv="pareto",  cross_rate=0.5,otype='a',functions="+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite", max_depth=3,   max_dim=10,  random_state=0, erc = False,  obj="fitness,complexity", shuffle=False,  split=0.75,  fb=0.5

