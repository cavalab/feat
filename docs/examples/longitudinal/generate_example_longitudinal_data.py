import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, rv_discrete
import pdb

def make_classification(pdict, data_long):
    """return a classification"""
    # if patient has a max glucose higher than a threshold and a bmi slope > 0, classify true
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
    feat1 = bmislope #> 0.0
    feat2 = glucmax #> 0.8
    # print('bmislope:',bmislope)
    model = (-1.4 + 1500*feat1 + 10.0*feat2)
    
    # classification = np.exp(model)/(1+np.exp(model)) > 0.5

    # print('feat1 value:',feat1,';',
    #       'feat2 value:',feat2,';',
    #       'model:',model,
    #       'classification:',classification)
    # return classification 
    return model
    # return 1 if bmislope > 0.0 and glucmax > 0.8 else 0

if __name__ == '__main__':
    np.random.seed(42)
    # generate data
    patients = np.arange(1000)
    measurements=['bmi','age','glucose']
    data=[]
    data_long=[]

    ptid = 0
    for p in patients:
        # tabular data
        pdict = {}
        pdict['id']=ptid
        pdict['sex']=np.random.randint(0,2)
        pdict['race']=np.random.randint(0,6)
        # data.append(pdict)
        # long data 
        age = np.random.randint(18,85)
        date = np.random.randint(1000,5000)

        p_visits = []
        for visit in np.arange(np.random.randint(1,100)):
                    
            age = age + np.random.randint(1,4) 
            date = date + np.random.randint(365,3*365) 
            
            for m in measurements:
                plongdict={}
                plongdict['id'] = ptid
                plongdict['name']=m
                plongdict['date']=date 
               
                if m == 'bmi':
                    plongdict['value'] = int(visit*np.random.randn()) + 40
                elif m == 'age':
                    plongdict['value'] = age
                elif m == 'glucose': 
                    plongdict['value'] = np.random.rand() 
                
                p_visits.append(plongdict)
     
        target = make_classification(pdict, p_visits) # np.random.randint(0,2)
        if not np.isnan(target):
            pdict['target'] = target
            data.append(pdict)
            data_long += p_visits
            ptid += 1

    df = pd.DataFrame.from_records(data, index='id', columns=['id','sex','race','target'])
    df_long = pd.DataFrame.from_records(data_long, index='id', columns=['id','name','date','value'])
    df.sort_index(axis=0,inplace=True)   
    df.to_csv('../data/d_example_patients.csv')
    print(np.sum(df['target']==0),'controls,',np.sum(df['target']==1),'cases')
    #shuffle rows
    df_long.to_csv('../data/d_example_patients_long.csv')
