from multiprocessing.dummy import Pool
import argparse
import FileUtils as fUtils
import numpy as np
from pmlb import fetch_data
from pmlb import dataset_names
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import feat as ft

#Default Parameter Values
split = 0.75 #train/test split
n = 10 #Number of iterations to run for the results
v = 0 #verbosity
features = "" 
labels = ""
dataset = "" #dataset name
datasetPath = ""  #dataset Folder Path
resultsPath = "" #path to the results

def customPrint(msg):
    if ( v > 0 ):
        print (msg)
        
def runFeat(iter):
    customPrint(  '[' +str(iter) + ']: Running Feat Classifier')
    features_train, features_test, labels_train, labels_test = train_test_split(features, label, train_size=split, test_size=1-split)
    
    customPrint( '[' +str(iter) + ']: Fitting the Regressor ') 
    
    fb =  [0.2, 0.5, 0.8]
    ml = ["LinearRidgeRegression","SVM","CART"]
    cross_rate = [0.2, 0.8]
        
    clf = GridSearchCV(ft.Feat( max_depth=5,max_dim=min(50,2*features.shape[1]),verbosity=v,shuffle=True),cv=5, param_grid= {"fb" : fb , "cross_rate" : cross_rate } ,verbose=0)
    clf.fit(features_train,labels_train)
    customPrint(  '[' +str(iter) + ']: Predicting the labels')
    labels_pred = clf.predict( features_test )
    customPrint  ('[' +str(iter) + '] MSE: ' + str(mse ( labels_test, labels_pred )))
    return mse ( labels_test, labels_pred )


def startParallelProcessing():
    global dataset,features,label
    # Setup a list of processes that we want to run
    features, label = fetch_data(dataset, return_X_y = True, local_cache_dir = datasetPath )
    
    customPrint ('Computing results for the dataset...'+dataset)
    p=Pool(n)
    results = p.map( runFeat, range(n)  )
    customPrint ('Writing results to csv file...')
    print (results)
    fUtils.writeToCSV( resultsPath + "/" +  dataset + ".csv",results,"feat-r",n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser",add_help=False)
    parser.add_argument('-split', action='store', dest='SPLIT',default=.75,type=float, help='Split Ratio for Train/Test Split')
    parser.add_argument('-n', action='store', dest='ITERATIONS',default=10,type=int, help='No Of Iterations')
    parser.add_argument('-verbose', action='store', dest='VERBOSE',default=0,type=int, help='0 for no verbose. 1 or 2 for verbose')
    parser.add_argument('-dataset', action='store', dest='DATASET',default="undefined",type=str, help='Input the Dataset Name')

    parser.add_argument('-dataset_path', action='store', dest='DATASET_PATH',default="undefined",type=str, help='Folder Path for the dataset')
    parser.add_argument('-results_path', action='store', dest='RESULTS_PATH',default="undefined",type=str, help='Path to write the results')
    
    args = parser.parse_args()

    split = args.SPLIT
    n = args.ITERATIONS
    v = args.VERBOSE
    dataset = args.DATASET
    datasetPath = args.DATASET_PATH
    resultsPath = args.RESULTS_PATH

    if ( v > 0 ):
        print('Paramters set...')
        print('split',split)
        print('n',n)
        print('verbose',v)
        print('dataset',dataset)
        print('dataset folder path:' , datasetPath)
        print('results path: ' , resultsPath)
        
    startParallelProcessing()
