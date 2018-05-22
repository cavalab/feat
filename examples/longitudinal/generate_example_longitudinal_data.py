import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, rv_discrete
np.random.seed(42)
# generate data
ptid = np.arange(20)
measurements=['bmi','age','glucose']
data=[]
data_long=[]
for p in ptid:
    pdict = {}
    pdict['id']=p
    pdict['sex']=np.random.randint(0,2)
    pdict['race']=np.random.randint(0,6)
    pdict['class']=np.random.randint(0,2)
    data.append(pdict)
    # long data 
    age = np.random.randint(18,85)
    date = np.random.randint(1000,5000)

    for visit in np.arange(np.random.randint(1,7)):
                
        age = age + np.random.randint(1,4) 
        date = date + np.random.randint(365,3*365) 
        
        for m in measurements:
            plongdict={}
            plongdict['id'] = p
            plongdict['name']=m
            plongdict['date']=date 
           
            if m == 'bmi':
                plongdict['value'] = int(10*np.random.randn()) + 40
            elif m == 'age':
                plongdict['value'] = age
            elif m == 'glucose': 
                plongdict['value'] = np.random.rand() 
            
            data_long.append(plongdict)
 
df = pd.DataFrame.from_records(data, index='id', columns=['id','sex','race','class'])
df_long = pd.DataFrame.from_records(data_long, index='id', columns=['id','name','date','value'])
df.sort_index(axis=0,inplace=True)   
df.to_csv('d_example_patients.csv')
#shuffle rows
df_long.to_csv('d_example_patients_long.csv')
