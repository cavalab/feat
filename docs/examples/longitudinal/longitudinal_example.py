import pandas as pd

import numpy as np

from feat import Feat
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('../data/d_example_patients.csv')
df.drop('id',axis=1,inplace=True)
X = df.drop('class',axis=1).values
y = df['class'].values
zfile = '../data/d_example_patients_long.csv'
kf = StratifiedKFold(n_splits=3)
kf.get_n_splits(X)


clf = Feat(max_depth=5,
           max_dim=min(50,2*X.shape[1]),
           gens = 20,
           pop_size = 100,
           verbosity=3,
           shuffle=True,
           ml='LR',
           classification=True,
           feature_names = ','.join(df.drop('class',axis=1).columns),
           functions="+,-,*,/,exp,log,and,or,not,=,<,<=,>,>=,ite,split,split_c,"
                     "mean,median,max,min,variance,skew,kurtosis,slope,count",
           backprop=True,
           iters=10,
           random_state=42,
           n_threads=1)
print('Feat version:',clf.__version__)
scores=[]

for train_idx, test_idx in kf.split(X,y):
    # print('train_idx:',train_idx)
    clf.fit(X[train_idx],y[train_idx],zfile,train_idx)
    scores.append(clf.score(X[test_idx],y[test_idx],zfile,test_idx))

print('scores:',scores)

###################################################################################################
# fit to all data
###################################################################################################

print('fitting longer to all data...')
clf.verbosity = 2
# clf.gens = 1000
# clf.pop_size = 100
# clf.max_stall = 20
clf.fit(X,y,zfile,np.arange(len(X)))
print('representation:\n',clf.get_representation())
print('model:\n',clf.get_model())

##################################################################################################
# Visualize the representation
###################################################################################################

Phi = clf.transform(X,zfile,np.arange(len(X)))
print('Phi:',Phi.shape,Phi)
# proj = np.vstack((Phi[:,0],Phi[:,2])).transpose()
proj = Phi
#scale the projection to zero mean, unit deviation
from sklearn.preprocessing import StandardScaler
proj = StandardScaler().fit_transform(proj)

print('proj:',proj.shape,proj)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects


# We choose a color palette with seaborn.
palette = np.array(sns.color_palette("cividis", 2))

# We create a scatter plot.
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(proj[:,0], proj[:,1], lw=0, s=20,
                c=palette[y.astype(np.int)])
ax.axis('square')
# ax.axis('off')
ax.axis('tight')

# We add the labels for each digit.
txts = []
for i in range(2):
    # Position of each label.
    xtext, ytext = np.median(proj[y == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)

# add labels from representation
rep = [r.split('[')[-1] for r in clf.get_representation().split(']') if r != '']
print('rep:',rep)
plt.xlabel(rep[0])
plt.ylabel(rep[1])

plt.savefig('longitudinal_representation.svg', dpi=120)
