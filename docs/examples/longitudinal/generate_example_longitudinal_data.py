import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, rv_discrete

def make_classification(pdict, data_long):
    """return a classification"""
    # if patient has a max glucose higher than 0.6 and a bmi slope > 0, classify true
    ld = [d for d in data_long if d['id'] == pdict['id']]
    bmis = []
    dates = []
    glucmax = 0
    for d in ld:
        if d['name'] == 'bmi':
            bmis.append(d['value'])
            dates.append(d['date'])
        elif d['name'] == 'glucose':
            glucmax = max(glucmax, d['value'])
    bmis = np.array(bmis)

    dates = np.array(dates)
    bmislope = np.cov(bmis,dates,ddof=0)[0,1]/np.var(dates) 
    return 1 if bmislope > 0.0 and glucmax > 0.8 else 0

if __name__ == '__main__':
    np.random.seed(42)
    # generate data
    ptid = np.arange(1000)
    measurements=['bmi','age','glucose']
    data=[]
    data_long=[]

    for p in ptid:
        # tabular data
        pdict = {}
        pdict['id']=p
        pdict['sex']=np.random.randint(0,2)
        pdict['race']=np.random.randint(0,6)
        data.append(pdict)
        # long data 
        age = np.random.randint(18,85)
        date = np.random.randint(1000,5000)

        for visit in np.arange(np.random.randint(1,20)):
                    
            age = age + np.random.randint(1,4) 
            date = date + np.random.randint(365,3*365) 
            
            for m in measurements:
                plongdict={}
                plongdict['id'] = p
                plongdict['name']=m
                plongdict['date']=date 
               
                if m == 'bmi':
                    plongdict['value'] = int(visit*np.random.randn()) + 40
                elif m == 'age':
                    plongdict['value'] = age
                elif m == 'glucose': 
                    plongdict['value'] = np.random.rand() 
                
                data_long.append(plongdict)
     
        pdict['class']= make_classification(pdict, data_long) # np.random.randint(0,2)
    df = pd.DataFrame.from_records(data, index='id', columns=['id','sex','race','class'])
    df_long = pd.DataFrame.from_records(data_long, index='id', columns=['id','name','date','value'])
    df.sort_index(axis=0,inplace=True)   
    df.to_csv('d_example_patients.csv')
    print(np.sum(df['class']==0),'controls,',np.sum(df['class']==1),'cases')
    #shuffle rows
    df_long.to_csv('d_example_patients_long.csv')
