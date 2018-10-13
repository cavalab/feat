import pandas as pd

import numpy as np

from feat import Feat
from sklearn.model_selection import KFold

df = pd.read_csv('d_example_patients.csv')
df.drop('id',axis=1,inplace=True)
X = df.drop('class',axis=1).values
y = df['class'].values
zfile = 'd_example_patients_long.csv'
kf = KFold(n_splits=3)
kf.get_n_splits(X)

clf = Feat(max_depth=5,
           max_dim=min(50,2*X.shape[1]),
           gens = 1,
           pop_size = 2,
           verbosity=1,
           shuffle=True,
           ml='LR',
           classification=True,
           functions="max,+,-,*,/,exp,log,and,or,not,=,<,>,ite,mean,median,min,variance,skew,kurtosis,slope,count",
           random_state=42)
scores=[]
for train_idx, test_idx in kf.split(X):
    clf.fit(X[train_idx],y[train_idx],zfile,train_idx)
    scores.append(clf.score(X[test_idx],y[test_idx],zfile,test_idx))

print('scores:',scores)
