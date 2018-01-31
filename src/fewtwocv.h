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
        		 bool cl = false,
        		 vector<int> popRange = vector<int>{100, 500},
        		 vector<int> genRange = vector<int>{100, 200},
        		 vector<float> fbRange = vector<float>{0.25, 0.5, 0.8},
        		 vector<float> crRange = vector<float>{0.25, 0.5, 0.75}
        		 );    
        
        void fit(MatrixXd x, VectorXd &y);
        
        VectorXd predict(MatrixXd x);
        
        void set_fold_size(int fdSize){ foldSize = fdSize; }
        
        void set_ml(vector<string> mlStr){ ml = mlStr; }
        
        void set_populations(vector<int> popRange){ populationRange = popRange; }
        
        void set_generations(vector<int> genRange){ generationRange = genRange; }
        
        void set_feedback(vector<float> fbRange){ feedbackRange = fbRange; }
        
        void set_cross_rates(vector<float> crRates){ crossRates = crRates; }
        
        private:
        
        void create_objects();
                
        void create_folds(int cols);
        
        int foldSize;
        bool classification;
        vector<string> ml;
        vector<int> populationRange;
        vector<int> generationRange;
        vector<float> feedbackRange;
        vector<float> crossRates;
        
        string functions;  
        vector<struct FewObjects> fewObjs;
        vector<struct DataFolds> dataFolds;
        int bestScoreIndex; 
    };
    
    FewtwoCV::FewtwoCV(int fdSize,
        			   bool cl,
        			   vector<int> popRange,
        		 	   vector<int> genRange,
        		 	   vector<float> fbRange,
        		 	   vector<float> crRange)
	{
		cout<<fdSize<<cl<<std::endl;
		
		foldSize = fdSize;
		classification = cl;
		populationRange = popRange;
		generationRange = genRange;
		feedbackRange = fbRange;
		crossRates = crRange;
		
		if(classification)
			ml = vector<string>{"LR", "CART"};
		else
			ml = vector<string>{"LinearRidgeRegression", "CART"};
    }

	void FewtwoCV::create_objects()
    {
    
    	cout<<"Creating objects\n";
    	    	
    	for(auto &curml : ml)
    		for(auto &curpop : populationRange)
    			for(auto &curgen : generationRange)
    				for(auto &curfb : feedbackRange)
						for(auto &curcr : crossRates)
						{
							struct FewObjects curobj;
							curobj.obj = Fewtwo(curpop,											//population size
    							   				curgen,											//generations
    							   				curml,											//ml string
    							   				classification,									//classification
    							   				0,												//verbosity
    							   				0,												//max_stall
    							   				"lexicase",										//selection
    							   				"pareto", 										//survivability
    							   				curcr,											//cross_rate
    							   				'a',											//otype
    							   				"+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite",	//nodes
    							   				3,												//dimension
    							   				10,												//depth 
    							   				0,												//random_state
    							   				false,											//erc
    							   				"fitness,complexity",							//objectives
    							   				false,											//shuffle
    							   				0.75,											//split
    							   				curfb);											//feedback
    							   				
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
    	
    	//cout<<"data size is "<<x.rows()<<"X"<<x.cols()<<"\n\n";
		//cout<<"x is \n"<<x<<"\n";
		
    	create_objects();
    	create_folds(x.cols());
    	
    	cout<<"Total number of objects created are "<<fewObjs.size()<<"\n";
    	
    	//#pragma omp parallel for
    	for(i = 0; i < fewObjs.size(); i++)
    	{
    		VectorXd objScore(foldSize);
    		
    		int testIndex = foldSize - 1;
    		
	    	for(j = 0; j < foldSize; j++)
	    	{
	    	
	    		cout<<"\n***************************\nRUNNING FOR i = "<<i<<" and j = "<<j<<"\n*************************************\n\n";
	    		
	    		int size = 0, filled = 0, k, l;
    	
				for(k = 0; k < foldSize; k++)
					if( k != testIndex)
						size += dataFolds[k].quantity;
			
				MatrixXd trainX(x.rows(), size);
				VectorXd trainY(size);
						
				for(k = 0; k < foldSize; k++)
				{
					if(k != testIndex)
					{
						//cout<<"k = "<<k<<" filled = "<<filled<<" dataFolds[k].quantity = "<<dataFolds[k].quantity<<"\n";
						trainX.block(0, filled, x.rows(), dataFolds[k].quantity) = x.block(0, dataFolds[k].startIndex, x.rows(), dataFolds[k].quantity);
						//trainY.block(filled, 0, 1, dataFolds[k].quantity) = y.block(dataFolds[k].startIndex, 0, 1, dataFolds[k].quantity);				
						filled += dataFolds[k].quantity;
					}
				}
								
				for(k = 0, l = 0; k < y.size(); k++)
					if(k < dataFolds[testIndex].startIndex || k >= dataFolds[testIndex].startIndex+dataFolds[testIndex].quantity)
						trainY(l++) = y(k);
				
		
				fewObjs[i].obj.set_dtypes(find_dtypes(trainX));
				fewObjs[i].obj.fit(trainX, trainY);
				
				MatrixXd testData(x.rows(), dataFolds[testIndex].quantity);
				VectorXd actualValues(dataFolds[testIndex].quantity);
		
				testData << x.block(0, dataFolds[testIndex].startIndex, x.rows(), dataFolds[testIndex].quantity);
				
				for(k = 0; k < dataFolds[testIndex].quantity; k++)
					actualValues(k) = y(dataFolds[testIndex].startIndex+k);
				//actualValues << y.block(dataFolds[testIndex].startIndex, 0, 1, dataFolds[testIndex].quantity);
				
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
	    
	    //cout<<"Hello";
	    bestScoreIndex = 0;
	    
	    for(i = 1; i < fewObjs.size(); i++)
	    	if(fewObjs[i].score < fewObjs[bestScoreIndex].score)
	    		bestScoreIndex = i;
	    		
	    cout << "\n********************\nBest tuning parameters for this data is \n";
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
