/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FEWTWOCV_H
#define FEWTWOCV_H

#include "feat.h"
namespace FT{

    /*!
     * @class FeatCV
     * @brief cross validator wrapper. 
     */
     
    struct BestParams
    {
    	string ml;
    	int population;
    	int generations;
    	int dimensions;
    	int depth;
    };
    
    class FeatCV
    {
    
    	public:
    	
        FeatCV(vector<string> mlStr, int popRange = 100, int genRange = 100, int dimRange = 10, int depRange = 3);    
            
        void create_objects();
                
        void create_folds(MatrixXd &x, VectorXd &y);
        
        vector<char> fold_dtypes(MatrixXd &X);
        
        void validate_data(MatrixXd x, VectorXd &y);
        
        private:
        
        int populationRange;
        int generationRange;
        int dimensionRange;
        int depthRange;
        vector<string> ml;
        vector<Feat> fewObjs;
        vector<double> scores;
        vector<struct BestParams> params;
        vector<MatrixXd> dataFolds;
        vector<VectorXd> outputFolds;
        
    };
    
    FeatCV::FeatCV(vector<string> mlStr, int popRange, int genRange, int dimRange, int depRange)
    {
    	ml = mlStr;
    	populationRange = popRange;
        generationRange = genRange;
        dimensionRange = dimRange;
    	depthRange = depRange;
    }

	void FeatCV::create_objects()
    {
    
    	cout<<"Creating objects\n";
    	
    	int curMl;
    	int curpop;
    	int curgen;
    	int curdim;
    	int curdep;
    	
    	for(curMl = 0; curMl < ml.size(); curMl++)
    		for(curpop = 1; curpop < populationRange; curpop++)
    			for(curgen = 1; curgen < generationRange; curgen++)
    				for(curdim = 1; curdim < dimensionRange; curdim++)
    					for(curdep = 1; curdep < depthRange; curdep++)
	    				{
	    					fewObjs.push_back(
	    						   Feat(curpop,	//population size
	    							   curgen,		//generations
	    							   ml[curMl],	//ml string
	    							   false,		//classification
	    							   1,			//verbosity
	    							   0,			//max_stall
	    							   "lexicase",	//selection
	    							   "pareto", 	//survivability
	    							   0.5,			//cross_rate		TODO : check if this is to be used in SVM
	    							   'a',			//otype
	    							   "+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite",	//nodes
	    							   curdim,		//dimension
	    							   curdep));	//depth
	    					
	    					struct BestParams obj;
	    					
	    					obj.ml = curMl;
							obj.population = curpop;
							obj.generations = curgen;
							obj.dimensions = curdim;
							obj.depth = curdep;
							
	    					params.push_back(obj);
	    							   
	    				}
	    				
	    cout<<"Objects creation complete\n";
    }
    
    void FeatCV::create_folds(MatrixXd &x, VectorXd &y)
    {
    
    	cout<<"Creating folds\n";
    	
    	int quantity = x.cols()/5;
    	int remains = x.cols()%5;
    	int i;
    	int startIndex = 0;
    	
    	//cout<<"Input data size is "<<x.rows()<<"X"<<x.cols()<<"\n\n";
    	//cout<<"Input output size is "<<y.size()<<"\n\n";
    	
    	//cout<<"Output split "<< y.block(0, 0, 1, 5)<<"\n";
    	//cout<<"Output split "<< y.block(0, startIndex, 1, quantity+1)<<"\n";
    	
    	for(i = 0; i < 5; i++)
    	{
    		if(i+1 <= remains)
    		{
    			dataFolds.push_back(x.block(0, startIndex, x.rows(), quantity+1));
    			VectorXd out(quantity+1);
    			out << y.block(startIndex, 0, 1, quantity+1);
    			outputFolds.push_back(out);
    			startIndex += quantity+1;
    		}
    		else
    		{
    			dataFolds.push_back(x.block(0, startIndex, x.rows(), quantity));
    			VectorXd out(quantity);
    			out << y.block(startIndex, 0, 1, quantity);
    			outputFolds.push_back(out);
    			startIndex += quantity;
    		}        		
    	}
    	
    	/*cout<<"Folds are\n";
    	for(i = 0; i < 5; i++)
    	{
    		cout<<"DataFold "<<i+1<<" size is "<<dataFolds[i].rows()<<"X"<<dataFolds[i].cols()<<"\n";
    		cout<<"OutputFold "<<i+1<<" size is "<<outputFolds[i].size()<<"\n\n";
    		cout<<"Data Fold "<<i+1<<" is \n"<<dataFolds[i]<<std::endl;
    		cout<<"Output Fold "<<i+1<<" is \n"<<outputFolds[i]<<std::endl;
    	}*/
    	
    	cout<<"Folds creation complete\n";
    }
    
    vector<char> FeatCV::fold_dtypes(MatrixXd &X)
    {
    	int i, j;
	    bool isBinary;
	    
	    vector<char> dtypes;
	    
	    for(i = 0; i < X.rows(); i++)
	    {
	        isBinary = true;
	        //cout<<"Checking for column "<<i<<std::endl;
	        for(j = 0; j < X.cols(); j++)
	        {
	            //cout<<"Value is "<<X(i, j)<<std::endl;
	            if(X(i, j) != 0 && X(i, j) != 1)
	                isBinary = false;
	        }
	        if(isBinary)
	            dtypes.push_back('b');
	        else
	            dtypes.push_back('f');
	    }
	    
	    return dtypes;
    }
    
    void FeatCV::validate_data(MatrixXd x, VectorXd &y)
    {
            	
    	int i, j, train1 = 0, train2 = 1, train3 = 2, train4 = 3, testIndex = 4;
    	
    	int bestScoreIndex = 0;
    	
    	//create_objects();
    	create_folds(x, y);
    	
    	MatrixXd trainX(dataFolds[0].rows(),
    					dataFolds[train1].cols() +
	    				dataFolds[train2].cols() + 
	    				dataFolds[train3].cols() + 
	    				dataFolds[train4].cols());
						
		trainX << dataFolds[train1],
				  dataFolds[train2], 
				  dataFolds[train3], 
				  dataFolds[train4];
				  
		VectorXd trainY(outputFolds[train1].size() + 
						outputFolds[train2].size() + 
						outputFolds[train3].size() + 
						outputFolds[train4].size());
						
		trainY << outputFolds[train1],
				  outputFolds[train2],
				  outputFolds[train3],
				  outputFolds[train4];
				  
		cout<<"Input data size is "<<trainX.rows()<<"X"<<trainX.cols()<<"\n\n";
		cout<<"trainx is \n"<<trainX<<"\n";
		cout<<"trainy is \n"<<trainY<<"\n";
		
		fewObjs.push_back(
	    				  Feat(100,	//population size
	    				  100,		//generations
	    				  "LinearRidgeRegression",	//ml string
	    				  false,		//classification
	    				  1,			//verbosity
	    				  0,			//max_stall
	    				  "lexicase",	//selection
	    				  "pareto", 	//survivability
	    				  0.5,			//cross_rate		TODO : check if this is to be used in SVM
	    				  'a',			//otype
	    				  "+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite",	//nodes
	    				  3,		//dimension
	    				  10));	//depth
	    							   
	    i = 0;
				  
		fewObjs[i].set_dtypes(fold_dtypes(trainX));
		
		fewObjs[i].fit(trainX, trainY);
		
		cout<<"Actual is \n"<<outputFolds[testIndex]<<"\n";
		
		cout<<"Prediction data is \n"<<dataFolds[testIndex]<<"\n";
		
		VectorXd prediction = fewObjs[i].predict(dataFolds[testIndex]);
		
		cout<<"Prediction is \n"<<prediction<<"\n";
		
		VectorXd objScore(5);
		
		if (fewObjs[i].get_classification())  // use classification accuracy
			objScore[j] = ((prediction.cast<int>().array() != outputFolds[testIndex].cast<int>().array()).cast<double>()).mean();
		else                        // use mean squared error
			objScore[j] = ((prediction - outputFolds[testIndex]).array().pow(2)).mean();
			
		cout<<"Error score is "<<objScore.mean()<<"\n";
    	
    	
    	/*for(i = 0; i < fewObjs.size(); i++)
    	{
    		VectorXd objScore(5);
    		
	    	for(j = 0; j < 5; j++)
	    	{
	    		MatrixXd trainX(dataFolds[train1].rows() + 
	    						dataFolds[train2].rows() + 
	    						dataFolds[train3].rows() + 
	    						dataFolds[train4].rows(), 
	    						dataFolds[0].cols());
	    						
	    		trainX << dataFolds[train1].rows(),
	    				  dataFolds[train2].rows(), 
	    				  dataFolds[train3].rows(), 
	    				  dataFolds[train4].rows();
	    				  
	    		VectorXd trainY(outputFolds[train1].size() + 
	    						outputFolds[train2].size() + 
	    						outputFolds[train3].size() + 
	    						outputFolds[train4].size());
	    						
	    		trainY << outputFolds[train1],
	    				  outputFolds[train2],
	    				  outputFolds[train3],
	    				  outputFolds[train4];
	    				  
	    		fewObjs[i].set_dtypes(fold_dtypes(trainX));
	    		
	    		fewObjs[i].fit(trainX, trainY);
	    		
	    		VectorXd prediction = fewObjs[i].predict(dataFolds[testIndex]);
	    		
	    		if (fewObjs[i].get_classification())  // use classification accuracy
	    			objScore[j] = ((prediction.cast<int>().array() != outputFolds[testIndex].cast<int>().array()).cast<double>()).mean();
    			else                        // use mean squared error
        			objScore[j] = ((prediction - outputFolds[testIndex]).array().pow(2)).mean();
	    	}
	    	
	    	scores.push_back(objScore.mean());
	    	
	    	train1 = (train1+1)%5;
	    	train2 = (train2+1)%5;
	    	train3 = (train3+1)%5;
	    	train4 = (train4+1)%5;
	    	testIndex = (testIndex+1)%5;		    	
	    }
	    
	    for(i = 1; i < scores.size(); i++)
	    	if(scores[i] < scores[bestScoreIndex])
	    		bestScoreIndex = i;
	    		
	    cout << "Best tuning parameters for this data is \n";
	    cout << "ML = "<< params[bestScoreIndex].ml;
	    cout << "Population size = "<< params[bestScoreIndex].population;
	    cout << "Generations = "<< params[bestScoreIndex].generations;
	    cout << "Dimensions = "<< params[bestScoreIndex].dimensions;
	    cout << "Depth = "<< params[bestScoreIndex].depth;
	    */
    }
    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
}
#endif
