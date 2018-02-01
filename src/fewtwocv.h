/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FEWTWOCV_H
#define FEWTWOCV_H

#include "fewtwo.h"

namespace FT{         
    
    struct DataFolds
    {
    	int startIndex;
    	int quantity;
    };
    
    /*!
     * @class FewObjects
     * @brief Structure for fewtwo object and its validation score
     */
    struct FewObjects
    {
    	Fewtwo obj;
    	double score;
    };
    
    /*!
     * @class FewtwoCV
     * @brief cross validator wrapper. 
     */
    class FewtwoCV
    {
    
    	public:
    	
    	///constructor
        FewtwoCV(int fdSize = 5,
        		 bool cl = false,
        		 vector<int> popRange = vector<int>{100, 500},
        		 vector<int> genRange = vector<int>{100, 200},
        		 vector<float> fbRange = vector<float>{0.2, 0.5, 0.8},
        		 vector<float> crRange = vector<float>{0.25, 0.5, 0.75}
        		 );    
        
        /// fit method to find the best fewtwo object based on input range
        void fit(MatrixXd x, VectorXd &y);
        
        /// predict method to predict the values based on best fewtwo object identified
        VectorXd predict(MatrixXd x);
        
        /// set the number of folds in k-way cross validation
        void set_fold_size(int fdSize){ foldSize = fdSize; }
        
        /// set ml values for cross validation
        void set_ml(vector<string> mlStr){ ml = mlStr; }
        
        /// set population sizes for cross validation
        void set_populations(vector<int> popRange){ populationRange = popRange; }
        
        /// set generations for cross validation
        void set_generations(vector<int> genRange){ generationRange = genRange; }
        
        /// set feedback values for cross validation
        void set_feedback(vector<float> fbRange){ feedbackRange = fbRange; }
        
        /// set cross rate values for cross validation
        void set_cross_rates(vector<float> crRates){ crossRates = crRates; }
        
        private:
        
        /// creates a vector of all fewtwo objects for cross validation
        void create_objects();
                
        /// create indexes for different folds
        void create_folds(int cols);
        
        int foldSize;						///< fold size for k-mean cross validation
        bool classification;				///< whether classification ML methods are used or not
        vector<string> ml;					///< vector containing list of ML methods to use for cross validation
        vector<int> populationRange;		///< vector containg list of population values
        vector<int> generationRange;		///< vector containing list of generation values
        vector<float> feedbackRange;		///< vector containing list of feedback values
        vector<float> crossRates;			///< vector containing list of cross rates
          
        vector<struct FewObjects> fewObjs;	///< vector containing fewtwo objects to be used for cross validation
        vector<struct DataFolds> dataFolds;	///< vector containg data fold indexes
        int bestScoreIndex; 				///< index of the best fewtwo object
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
							curobj.obj. = set_ml(curml);
							curobj.obj. = set_pop_size(curpop);
							curobj.obj. = set_generations(curgen);
							curobj.obj. = set_feedback(curfb);
							curobj.obj. = set_cross_rate(curcr);
								   				
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
    	create_objects();
    	create_folds(x.cols());
    	
    	cout<<"Total number of objects created are "<<fewObjs.size()<<"\n";
    	
    	#pragma omp parallel for
    	for(int i = 0; i < fewObjs.size(); i++)
    	{
    		VectorXd objScore(foldSize);
    		
    		int testIndex = foldSize - 1;
    		
	    	for(int j = 0; j < foldSize; j++)
	    	{	    		
	    		int filled = 0, k, l;
			
				MatrixXd trainX(x.rows(), x.cols() - dataFolds[testIndex].quantity);
				VectorXd trainY(x.cols() - dataFolds[testIndex].quantity);
						
				for(k = 0; k < foldSize; k++)
				{
					if(k != testIndex)
					{
						trainX.block(0, filled, x.rows(), dataFolds[k].quantity) = 
									x.block(0, dataFolds[k].startIndex, x.rows(), dataFolds[k].quantity);
						filled += dataFolds[k].quantity;
					}
				}
								
				for(k = 0, l = 0; k < y.size(); k++)
					if(k < dataFolds[testIndex].startIndex ||
					   k >= dataFolds[testIndex].startIndex+dataFolds[testIndex].quantity)
						trainY(l++) = y(k);
				
		
				fewObjs[i].obj.set_dtypes(find_dtypes(trainX));
				fewObjs[i].obj.fit(trainX, trainY);
				
				MatrixXd testData(x.rows(), dataFolds[testIndex].quantity);
				VectorXd actualValues(dataFolds[testIndex].quantity);
		
				testData << 
					x.block(0, dataFolds[testIndex].startIndex, x.rows(), dataFolds[testIndex].quantity);
				
				for(k = 0; k < dataFolds[testIndex].quantity; k++)
					actualValues(k) = y(dataFolds[testIndex].startIndex+k);
				
				VectorXd prediction = fewObjs[i].obj.predict(testData);
		
				if (fewObjs[i].obj.get_classification())  	// use classification accuracy
					objScore[j] = ((prediction.cast<int>().array() != 
									actualValues.cast<int>().array()).cast<double>()).mean();
				else                        			// use mean squared error
					objScore[j] = ((prediction - actualValues).array().pow(2)).mean();
			
	    		testIndex = (testIndex+1)%foldSize;
	    		
	    	}
	    	
	    	fewObjs[i].score = objScore.mean();
	    	cout<<"****Finished for i = "<<i<<"\n";
	    	
	    }
	    
	    bestScoreIndex = 0;
	    
	    for(int i = 1; i < fewObjs.size(); i++)
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
