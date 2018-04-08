import argparse
import pandas as pd 
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def createBoxPlot(path,v,spath):
    print ('Getting all the files in the path...',path)
    print ('path...',path+"*.csv")
    csvs = glob.glob(path+"*")
    i = 0
    for csv in csvs:
      csv = csv.encode('utf-8')
      print ('Creating Box Plot for the file...',csv)
      
      df = pd.read_csv(csv.decode('utf-8'),encoding='utf8' ,header = 0,index_col=0)#.drop("Classifier",axis=1,inplace=True)
      if ( df.shape[0] == 2 ):
             continue #Feat hasn't been calcualted for this dataset yet
      df.T.boxplot()
      
      fname =  csv.replace( b".csv", b"" ) #file_name
      fname = fname[  fname.rfind(b'/') + 1 : ].decode('utf-8')
      plt.title(fname) 
      plt.ylabel('mean squared error')
      plt.xlabel('Classifiers')
      plt.savefig( spath + fname  )
      plt.clf() 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument Parser",add_help=False)
    parser.add_argument('-v', action='store', dest='VERBOSE',default=1,type=int, help='verbose')
    parser.add_argument('-p', action='store', dest='PATH',default=".",type=str, help='Path of the csv files')
    parser.add_argument('-s', action='store', dest='STOREPATH',default=".",type=str, help='Path to store the box plots')
    args = parser.parse_args()
    
    path = args.PATH
    v = args.VERBOSE
    spath = args.STOREPATH

    if ( v > 0 ):
        print('Paramters set...')
        print('Verbose...',v)
        print('Path...',path)
        print('Path to Store the box plots...',spath)

    createBoxPlot(path,v,spath)





