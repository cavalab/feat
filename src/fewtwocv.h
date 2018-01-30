/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FEWTWOCV_H
#define FEWTWOCV_H

#include "fewtwo.h"

namespace FT{

    /*!
     * @class FewtwoCV
     * @brief cross validator wrapper. 
     */
         
    struct DataFolds
    {
    	int startIndex;
    	int quantity;
    };
    
    struct FewObjects
    {
    	Fewtwo obj;
    	double score;
    };
    
    class FewtwoCV
    {
    
    	public:
    	
        FewtwoCV(int fdSize = 5,
        		 vector<string> mlStr = vector<string>{"LinearRidgeRegression"},
        		 vector<int> popRange = vector<int>{50, 100, 150, 200},
        		 vector<int> genRange = vector<int>{50, 100, 150, 200},
        		 vector<int> dimRange = vector<int>{1},
        		 vector<int> depRange = vector<int>{2},
        		 vector<float> crRange = vector<float>{0.5},
        		 string funcs = "+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite");    
        
        void fit(MatrixXd x, VectorXd &y);
        
        VectorXd predict(MatrixXd x);
        
        void set_ml(vector<string> mlStr){ ml = mlStr; }
        
        void set_populations(vector<int> popRange){ populationRange = popRange; }
        
        void set_generations(vector<int> genRange){ generationRange = genRange; }
        
        void set_dimensions(vector<int> dimRange){ dimensionRange = dimRange; }
        
        void set_depths(vector<int> depRange){ depthRange = depRange; }
        
        void set_cross_rates(vector<float> crRates){ crossRates = crRates; }
        
        void set_functions(string funcs) { functions = funcs; }
        
        private:
        
        void create_objects();
                
        void create_folds(int cols);
        
        int foldSize;
        vector<string> ml;
        vector<int> populationRange;
        vector<int> generationRange;
        vector<int> dimensionRange;
        vector<int> depthRange;
        vector<float> crossRates;
        string functions;  
        vector<struct FewObjects> fewObjs;
        vector<struct DataFolds> dataFolds;
        int bestScoreIndex; 
    };
    
    FewtwoCV::FewtwoCV(int fdSize,
    				   vector<string> mlStr,
    				   vector<int> popRange,
        		 	   vector<int> genRange,
        		 	   vector<int> dimRange,
        		 	   vector<int> depRange,
        		 	   vector<float> crRates,
        		 	   string funcs):
        		 	   foldSize(fdSize),
        		 	   ml(mlStr),
        		 	   populationRange(popRange),
        		 	   generationRange(genRange),
        		 	   dimensionRange(dimRange),
        		 	   depthRange(depRange),
        		 	   crossRates(crRates),
        		 	   functions(funcs),
        		 	   bestScoreIndex(-1){}

	void FewtwoCV::create_objects()
    {
    
    	cout<<"Creating objects\n";
    	
    	int curMl;
    	int curpop;
    	int curgen;
    	int curdim;
    	int curdep;
    	
    	for(auto &curMl : ml)
    		for(auto &curpop : populationRange)
    			for(auto &curgen : generationRange)
    				for(auto &curdim : dimensionRange)
    					for(auto &curdep : depthRange)
    						for(auto &curcr : crossRates)
    						{
    							struct FewObjects curobj;
    							curobj.obj = Fewtwo(curpop,		//population size
	    							   				curgen,		//generations
	    							   				curMl,		//ml string
	    							   				false,		//classification
	    							   				1,			//verbosity
	    							   				0,			//max_stall
	    							   				"lexicase",	//selection
	    							   				"pareto", 	//survivability
	    							   				curcr,		//cross_rate
	    							   				'a',		//otype
	    							   				functions,	//nodes
	    							   				curdim,		//dimension
	    							   				curdep);	//depth 
	    							   				
	    						curobj.score = 0;
	    						
	    						fewObjs.push_back(curobj);
    						}
	    cout<<"Objects creation complete\n";
    }
    
    void FewtwoCV::create_folds(int cols)
    {
    
    	cout<<"Creating folds\n";
    	
    	int quantity = cols/foldSize;
    	int remains = cols%foldSize;
    	int i;
    	int startIndex = 0;
    	
    	for(i = 0; i < foldSize; i++)
    	{
    		struct DataFolds fold;
    		
    		fold.startIndex = startIndex;
    		
    		if(i+1 <= remains)
    		{
    			fold.quantity = quantity+1;
    			startIndex += quantity+1;
    		}
    		else
    		{
    			fold.quantity = quantity;
    			startIndex += quantity;
    		}
    			
    		dataFolds.push_back(fold);
    		
    	}
    	
    	cout<<"Folds creation complete\n";
    }
    
    void FewtwoCV::fit(MatrixXd x, VectorXd &y)
    {
            	
    	int i, j;
    	
    	cout<<"data size is "<<x.rows()<<"X"<<x.cols()<<"\n\n";
		cout<<"x is \n"<<x<<"\n";
		
    	create_objects();
    	create_folds(x.cols());
    	
    	cout<<"Total number of objects created are "<<fewObjs.size()<<"\n";
    	for(i = 0; i < fewObjs.size(); i++)
    	{
    		VectorXd objScore(foldSize);
    		
    		int testIndex = foldSize - 1;
    		
	    	for(j = 0; j < foldSize; j++)
	    	{
	    	
	    		cout<<"\n***************************\nRUNNING FOR i = "<<i<<" and j = "<<j<<"\n*************************************\n\n";
	    		
	    		int size = 0, filled = 0, k;
    	
				for(k = 0; k < foldSize; k++)
					if( k != testIndex)
						size += dataFolds[k].quantity;
			
				MatrixXd trainX(x.rows(), size);
				VectorXd trainY(size);
		
				for(k = 0; k < foldSize; k++)
				{
					if(k != testIndex)
					{
						trainX.block(0, filled, x.rows(), dataFolds[k].quantity) = x.block(0, dataFolds[k].startIndex, x.rows(), dataFolds[k].quantity);
						trainY.block(filled, 0, 1, dataFolds[k].quantity) = y.block(dataFolds[k].startIndex, 0, 1, dataFolds[k].quantity);				
						filled += dataFolds[k].quantity;
					}
				}
		
				fewObjs[i].obj.set_dtypes(find_dtypes(trainX));
				fewObjs[i].obj.fit(trainX, trainY);
				
				MatrixXd testData(x.rows(), dataFolds[testIndex].quantity);
				VectorXd actualValues(dataFolds[testIndex].quantity);
		
				testData << x.block(0, dataFolds[testIndex].startIndex, x.rows(), dataFolds[testIndex].quantity);
				actualValues << y.block(dataFolds[testIndex].startIndex, 0, 1, dataFolds[testIndex].quantity);
				
				VectorXd prediction = fewObjs[i].obj.predict(testData);
		
				VectorXd objScore(foldSize);
		
				if (fewObjs[i].obj.get_classification())  	// use classification accuracy
					objScore[j] = ((prediction.cast<int>().array() != actualValues.cast<int>().array()).cast<double>()).mean();
				else                        			// use mean squared error
					objScore[j] = ((prediction - actualValues).array().pow(2)).mean();
			
	    		testIndex = (testIndex+1)%foldSize;		
	    	}
	    	
	    	fewObjs[i].score = objScore.mean();
	    	
	    }
	    
	    bestScoreIndex = 0;
	    
	    for(i = 1; i < fewObjs.size(); i++)
	    	if(fewObjs[i].score < fewObjs[bestScoreIndex].score)
	    		bestScoreIndex = i;
	    		
	    cout << "Best tuning parameters for this data is \n";
	    cout << "\nML = " << fewObjs[bestScoreIndex].obj.get_ml();
	    cout << "\nPopulation size = " << fewObjs[bestScoreIndex].obj.get_pop_size();
	    cout << "\nGenerations = " << fewObjs[bestScoreIndex].obj.get_generations();
	    cout << "\nDimensions = " << fewObjs[bestScoreIndex].obj.get_max_dim();
	    cout << "\nDepth = " << fewObjs[bestScoreIndex].obj.get_max_depth();
	    cout << "\nCross Rate = " <<fewObjs[bestScoreIndex].obj.get_cross_rate();
	    cout << "\nScore = " << fewObjs[bestScoreIndex].score<<"\n";
    }
    
    VectorXd FewtwoCV::predict(MatrixXd x)
    {
    	if(bestScoreIndex == -1)
    	{
    		std::cerr << "You need to call fit first to cross validate first.\n";
    		throw;
    	}
    	
    	return fewObjs[bestScoreIndex].obj.predict(x);
    }    
}
#endif
