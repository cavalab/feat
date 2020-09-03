# Using the Archive

In this example, we apply Feat to a regression problem and visualize the archive of representations. 

*Note: this code uses the [Penn ML Benchmark Suite](https://github.com/EpistasisLab/penn-ml-benchmarks/) to fetch data. You can install it using `pip install pmlb`.*

Also available as a [notebook](http://github.com/lacava/feat/blob/master/docs/examples/archive.ipynb)

## Training Feat

First, we import the data and create a train-test split.

```python
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np

dataset='690_visualizing_galaxy'
X, y = fetch_data(dataset,return_X_y=True)
X_t,X_v, y_t, y_v = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=42)
```

Then we set up a Feat instance and train the model, storing the final archive. 

```python
from feat import Feat
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
dataset='690_visualizing_galaxy'
X, y = fetch_data(dataset,return_X_y=True)
X_t,X_v, y_t, y_v = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=42)

# fix the random state
random_state=11314

fest = Feat(pop_size=500,                                                                        
            gens=100,                  # maximum of 200 generations                            
            max_time=60,              # max time of 60 seconds                                                         
            ml = "LinearRidgeRegression",   # use ridge regression (the default)                                                     
            sel='lexicase',                 # use epsilon lexicase selection (the default)                                                                surv='nsga2',                   # use nsga-2 survival (the defaut)                                                    
            max_depth=6,                    # constrain features to depth of 6                                                     
            max_dim=min([X.shape[1]*2,50]), # constrain representation dimensionality to 2x the feature space or 50                                                     
            random_state=random_state,                                                            
            hillclimb=True,                 # use stochastic hillclimbing to optimize weights for 10 iterations                                                  
            iters=10,                                                                            
            n_threads=1,                   # restricts to single thread                                                      
            verbosity=2,                   # verbose output                                                      
            tune_final=False               # don't tune final model
           ) 

print('FEAT version:', fest.__version__)
# train the model
fest.fit(X_t,y_t)

# get the test score
test_score = {}
test_score['feat'] = mse(y_v,fest.predict(X_v))

# store the archive
archive = fest.get_archive(justfront=True)

print('complexity','fitness','validation fitness',
     'eqn')
order = np.argsort([a['complexity'] for a in archive])
complexity = []
fit_train = []
fit_test = []
eqn = []

for o in order:
    model = archive[o]
    if model['pareto_rank'] == 1:
        print(model['complexity'],
              model['fitness'],
              model['fitness_v'],
              model['eqn'],
             )

        complexity.append(model['complexity'])
        fit_train.append(model['fitness'])
        fit_test.append(model['fitness_v'])
        eqn.append(model['eqn'])
```

For comparison, we can fit an Elastic Net and Random Forest regression model to the same data:

```python
# random forest
rf = RandomForestRegressor(random_state=random_state)

rf.fit(X_t,y_t)

test_score['rf'] = mse(y_v,rf.predict(X_v))

# elastic net
linest = ElasticNet()

linest.fit(X_t,y_t)

# test_score={}

test_score['elasticnet'] = mse(y_v,linest.predict(X_v))

```

## Visualizing the Archive

Let's visualize this archive with the test scores. This gives us a sense of how increasing the representation 
complexity affects the quality of the model and its generalization. 

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

matplotlib.rcParams['figure.figsize'] = (10, 6)
%matplotlib inline 
sns.set_style('white')
h = plt.figure(figsize=(14,8))

# plot archive points 
plt.plot(fit_train,complexity,'--ro',label='Train',markersize=6)
plt.plot(fit_test,complexity,'--bx',label='Validation')
# some models to point out
best = np.argmin(np.array(fit_test))
middle = np.argmin(np.abs(np.array(fit_test[:best])-test_score['rf']))
small = np.argmin(np.abs(np.array(fit_test[:middle])-test_score['elasticnet']))

print('best:',complexity[best])
print('middle:',complexity[middle])
print('small:',complexity[small])
plt.plot(fit_test[best],complexity[best],'sk',markersize=16,markerfacecolor='none',label='Model Selection')

# test score lines
y1 = -1
y2 = np.max(complexity)+1
plt.plot((test_score['feat'],test_score['feat']),(y1,y2),'--k',label='FEAT Test',alpha=0.5)
plt.plot((test_score['rf'],test_score['rf']),(y1,y2),'-.xg',label='RF Test',alpha=0.5)
plt.plot((test_score['elasticnet'],test_score['elasticnet']),(y1,y2),'-sm',label='ElasticNet Test',alpha=0.5)

print('complexity',complexity)
# eqn[best] = '0)]$\n$'.join(eqn[best].split('0)]'))
xoff = 10
for e,t,c in zip(eqn,fit_test,complexity):
#     if c in [1,4,12,31,43,53]:
    if c in [complexity[best],complexity[middle],complexity[small]]:
        t = t+xoff
        tax = plt.text(t,c,'$\leftarrow'+e+'$',size=18,horizontalalignment='left',
                      verticalalignment='center')
#         tax.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))

l = plt.legend(prop={'size': 16},loc=[1.01,0.25])
plt.xlabel('MSE',size=16)
plt.xlim(np.min(fit_train)*.75,np.max(fit_test)*2)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')

# plt.ylim(y1,y2)
plt.gca().set_yticklabels('')
plt.gca().set_xticklabels('')

plt.ylabel('Complexity',size=18)
h.tight_layout()
h.savefig('archive_example.svg')

plt.show()
```

This produces the figure below. 
Note that ElasticNet produces a similar test score to the linear representation
in Feat's archive, and that Random Forest's test score is near the representation shown in the middle.

The best model, marked with a square, is selected from the validation curve (blue line). 
The validation curve shows how models begin to overfit as complexity grows. 
By visualizing the archive, we can see that some lower complexity models achieve nearly as good of a validation score. 
In this case it may be preferable to choose that representation instead. 

![Feat archive](archive_example.svg)

