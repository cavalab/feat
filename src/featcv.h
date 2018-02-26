/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef FEATCV_H
#define FEATCV_H

#include "feat.h"

namespace FT{         
    
    /*!
     * @class DataFolds
     * @brief structure contains indexes and number of data sets in each fold
     */
    struct DataFolds
    {
    	int startIndex;
    	int quantity;
    };
    
    /*!
     * @class FewObjects
     * @brief Structure for feat object and its validation score
     */
    struct FeatObjects
    {
    	Feat obj;
    	double score;
    };
    
    enum tokenType{
    population,
    generation,
    ml,
    maxStall,
    selection,
    survival,
    crossRate,
    functions,
    maxDepth,
    maxDim,
    erc,
    objectives,
    feedBack,
    unknown
    };
    
    enum vecType{
    Integer,
    UInteger,
    Float,
    Double,
    String,
    Bool
    };
    
    bool string2bool (const std::string & v)
    {
        return !v.empty () &&
            (strcasecmp (v.c_str (), "true") == 0 ||
             atoi (v.c_str ()) != 0);
    }
    
    tokenType getTokenType(string &token)
    {
        if(!token.compare("pop_size"))
            return population;
        else if(!token.compare("generations"))
            return generation;
        else if(!token.compare("ml"))
            return ml;
        else if(!token.compare("max_stall"))
            return maxStall;
        else if(!token.compare("selection"))
            return selection;
        else if(!token.compare("survival"))
            return survival;
        else if(!token.compare("cross_rate"))
            return crossRate;
        else if(!token.compare("functions"))
            return functions;
        else if(!token.compare("max_depth"))
            return maxDepth;
        else if(!token.compare("max_dim"))
            return maxDim;
        else if(!token.compare("erc"))
            return erc;
        else if(!token.compare("objectives"))
            return objectives;
        else if(!token.compare("feedback"))
            return feedBack;
        else
            return unknown;
            
    }
    
    /*!
     * @class FeatCV
     * @brief cross validator wrapper. 
     */
    class FeatCV
    {
    
    	public:
    	 
        FeatCV(int fdSize, string params); 
        
        /// fit method to find the best feat object based on input range
        void fit(MatrixXd x, VectorXd &y);
        
        /// predict method to predict the values based on best feat object identified
        VectorXd predict(MatrixXd x);
        
        private:
        
        /// parse method to parse hyperParams from string and create feat objects
        void parse();
        
        /// push feat objects according to current set of hyperParams
        void pushObjects();
                
        /// create indexes for different folds
        void create_folds(int cols);
        
        /// clears all vectors for parameters
        void clearVectors();
        
        /// set default values in vectors if they are empty
        void setDefaults();
        
        /// fill a vector from a comma seperated string (string)
        void fillVector(vector<string> &vec, string &str);

        /// fill a vector from a comma seperated string (float/double)
        template<class T>
        void fillVector(vector<T> &vec, string &str, vecType type);
        
        
        int foldSize;                           ///< fold size for k-mean cross validation
        string hyperParams;                     ///< string containing parameters for cross validation
        
        vector<int> pop_size;
        vector<int> gens;
        vector<string> mlStr;
        vector<int> max_stall;
        vector<string> sel;
        vector<string> surv;
        vector<float> cross_rate;
        vector<string> funcs;
        vector<unsigned int> max_depth;
        vector<unsigned int> max_dim;
        vector<bool> ercVec;
        vector<string> obj;
        vector<double> fb;
          
        vector<struct FeatObjects> featObjs;    ///< vector list of feat objects for cross validation
        vector<struct DataFolds> dataFolds;     ///< vector containg data fold indexes
        int bestScoreIndex;                     ///< index of the best feat object
        
        
    };
    
    FeatCV::FeatCV(int fdSize, string params)
    {
        foldSize = fdSize;
        hyperParams = params;
    }
    
    void FeatCV::parse()
    {
        int startIndex = hyperParams.find('{');
        int endIndex = hyperParams.find("},");
        
        string curParams;
        string token, values;
        tokenType type;
        
        int tokenStart, tokenIndex, valueEnd, prevEnd; 
        
        while(startIndex != string::npos && endIndex != string::npos)
        {
            clearVectors();
            
            curParams = trim(hyperParams.substr(startIndex+1, endIndex - startIndex - 1));
            
            tokenStart = curParams.find('(');
            tokenIndex = curParams.find(':');
            valueEnd = curParams.find(')');
            
            while(tokenStart != string::npos && tokenIndex != string::npos && valueEnd != string::npos)
            {
                token = trim(curParams.substr(tokenStart + 2, tokenIndex - tokenStart - 3));
                values = trim(curParams.substr(tokenIndex + 1, valueEnd - tokenIndex - 1));
                //cout<<"\n\nCurParams = "<<curParams<<"\n";
                //cout<<"Token  = "<<token<<"\n";
                //cout<<"values = "<<values<<"\n";
                
                type = getTokenType(token);
                
                switch(type)
                {
                    case population :   fillVector(pop_size, values, Integer); break; 
                    case generation :   fillVector(gens, values, Integer); break;
                    case ml :           fillVector(mlStr, values); break;
                    case maxStall :     fillVector(max_stall, values, Integer); break;
                    case selection :    fillVector(sel, values); break;
                    case survival :     fillVector(surv, values); break;
                    case crossRate :    fillVector(cross_rate, values, Float); break;
                    case functions :    fillVector(funcs, values); break;
                    case maxDepth :     fillVector(max_depth, values, UInteger); break;
                    case maxDim :       fillVector(max_dim, values, UInteger); break;
                    case erc :          fillVector(ercVec, values, Bool); break;
                    case objectives :   fillVector(obj, values); break;
                    case feedBack :     fillVector(fb, values, Double); break;
                    case unknown:       cout<<"Unknown token type "<<token<<std::endl; throw;
                }
                
                tokenStart = curParams.find('(', valueEnd + 1);
                tokenIndex = curParams.find(':', valueEnd + 1);
                valueEnd = curParams.find(')', valueEnd + 1);
            }
            
            setDefaults();
            
            for(auto str : mlStr)
                cout<<"Ml method "<<str<<"\n";
            
            pushObjects();
            
            startIndex = hyperParams.find('{', endIndex + 1);
            endIndex = hyperParams.find("},", endIndex + 1);
            
        }
        
        clearVectors();
    }

	void FeatCV::pushObjects()
    {        
		for(auto &curPop : pop_size)
		for(auto &curGen : gens)
		for(auto &curMl : mlStr)
		for(auto &curMaxStall : max_stall)
		for(auto &curSel : sel)
		for(auto &curSurv : surv)
		for(auto &curCrossRate : cross_rate)
		for(auto &curFunc : funcs)
		for(auto &curMaxDepth : max_depth)
		for(auto &curMaxDim : max_dim)
		for(auto curErc : ercVec)
		for(auto &curObj : obj)
		for(auto &curFb : fb)
		{
		    struct FeatObjects curobj;
							
			curobj.obj.set_pop_size(curPop);
			curobj.obj.set_generations(curGen);
			curobj.obj.set_ml(curMl);
			curobj.obj.set_max_stall(curMaxStall);
			curobj.obj.set_selection(curSel);
			curobj.obj.set_survival(curSurv);
			curobj.obj.set_cross_rate(curCrossRate);
			curobj.obj.set_functions(curFunc);
			curobj.obj.set_max_depth(curMaxDepth);
			curobj.obj.set_max_dim(curMaxDim);
			curobj.obj.set_erc(curErc);
			curobj.obj.set_objectives(curObj);
			curobj.obj.set_feedback(curFb);
			
			curobj.obj.set_verbosity(0);
			curobj.score = 0;
			
			featObjs.push_back(curobj);   
		}  
    }
    
    void FeatCV::clearVectors()
    {
        pop_size.clear();
        gens.clear();
        mlStr.clear();
        max_stall.clear();
        sel.clear();
        surv.clear();
        cross_rate.clear();
        funcs.clear();
        max_depth.clear();
        max_dim.clear();
        ercVec.clear();
        obj.clear();
        fb.clear();
    }
    
    void FeatCV::setDefaults()
    {
        if(!pop_size.size())
             pop_size = vector<int>{100};
        if(!gens.size())
             gens = vector<int>{100};
        if(!mlStr.size())
             mlStr = vector<string>{"LinearRidgeRegression","CART"};
        if(!max_stall.size())
             max_stall = vector<int>{0};
        if(!sel.size())
             sel = vector<string>{"lexicase"};
        if(!surv.size())
             surv = vector<string>{"pareto"};
        if(!cross_rate.size())
             cross_rate = vector<float>{0.5};
        if(!funcs.size())
             funcs = vector<string>{"+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite"};
        if(!max_depth.size())
             max_depth = vector<unsigned int>{3};
        if(!max_dim.size())
             max_dim = vector<unsigned int>{10};
        if(!ercVec.size())
             ercVec = vector<bool>{false};
        if(!obj.size())
             obj = vector<string>{"fitness,complexity"};
        if(!fb.size())
             fb = vector<double>{0.5};
    }
    
    void FeatCV::fillVector(vector<string> &vec, string &str)
    {
        str += ",";          // add delimiter to end 
        string delim = "\",";
        size_t pos = 0, start;
        string token;
        vec.clear();
        while ((pos = str.find(delim)) != string::npos) 
        {
            start = str.find("\"");
            token = trim(str.substr(start+1, pos-start-1));
            vec.push_back(token);
            
            str.erase(0, pos + delim.length());
        }
    }
    
    template<class T>
    void FeatCV::fillVector(vector<T> &vec, string &str, vecType type)
    {
        str += ',';          // add delimiter to end 
        string delim = ",";
        size_t pos = 0;
        string token;
        vec.clear();
        //cout<<"Filling vector with str as "<<str<<"\n";
        while ((pos = str.find(delim)) != string::npos) 
        {
            token = trim(str.substr(0, pos));
            //cout<<"Pushed "<<token<<"\n";
            switch(type)
            {
                case Integer:   vec.push_back(stoi(token)); break;
                case UInteger:  vec.push_back(stoul(token)); break;
                case Float:     vec.push_back(atof(token.c_str())); break;
                case Double:    vec.push_back(atof(token.c_str())); break;
                case Bool:      vec.push_back(string2bool(token)); break;
            }
            //objectives.push_back(token);
            str.erase(0, pos + delim.length());
        }
    }
       
    void FeatCV::create_folds(int cols)
    {
    
    	cout<<"Creating folds\n";
    	
    	int quantity = cols/foldSize;
    	int remains = cols%foldSize;
    	int i;
    	int startIndex = 0;
    	
    	// determining indexes and umber of datasets in each fold
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
    
    void FeatCV::fit(MatrixXd x, VectorXd &y)
    {
    	parse();
    	create_folds(x.cols());
    	
    	cout<<"Total number of objects created are "<<featObjs.size()<<"\n";
    	
    	#pragma omp parallel for
    	for(int i = 0; i < featObjs.size(); i++)
    	{
    		VectorXd objScore(foldSize);
    		
    		int testIndex = foldSize - 1;
    		
    		// loop to fit all data folds in a single object and calculate the mean score
	    	for(int j = 0; j < foldSize; j++)
	    	{	    		
	    		int filled = 0, k, l;
			
				MatrixXd trainX(x.rows(), x.cols() - dataFolds[testIndex].quantity);
				VectorXd trainY(x.cols() - dataFolds[testIndex].quantity);
				
				//creating training data from data folds		
				for(k = 0; k < foldSize; k++)
				{
					if(k != testIndex)
					{
						trainX.block(0, filled, x.rows(), dataFolds[k].quantity) = 
							x.block(0, dataFolds[k].startIndex, x.rows(), dataFolds[k].quantity);
						filled += dataFolds[k].quantity;
					}
				}
				
				// creating training data outputs from data folds				
				for(k = 0, l = 0; k < y.size(); k++)
					if(k < dataFolds[testIndex].startIndex ||
					   k >= dataFolds[testIndex].startIndex+dataFolds[testIndex].quantity)
						trainY(l++) = y(k);
				
		        // set dtypes for training data and train the model
				featObjs[i].obj.set_dtypes(find_dtypes(trainX));
				featObjs[i].obj.fit(trainX, trainY);
				
				// extracting testdata and actual values for test data from data folds
				MatrixXd testData(x.rows(), dataFolds[testIndex].quantity);
				VectorXd actualValues(dataFolds[testIndex].quantity);
		
				testData << 
				x.block(0, dataFolds[testIndex].startIndex,x.rows(),dataFolds[testIndex].quantity);
				
				for(k = 0; k < dataFolds[testIndex].quantity; k++)
					actualValues(k) = y(dataFolds[testIndex].startIndex+k);
				
				// prediction on test data
				VectorXd prediction = featObjs[i].obj.predict(testData);
		        
		        // calculating difference between actual and predicted values
				if (featObjs[i].obj.get_classification())
				    // use classification accuracy
					objScore[j] = ((prediction.cast<int>().array() != 
							actualValues.cast<int>().array()).cast<double>()).mean();
				else
				    // use mean squared error
					objScore[j] = ((prediction - actualValues).array().pow(2)).mean();
			
	    		testIndex = (testIndex+1)%foldSize;
	    		
	    	}
	    	
	    	// setting mean score for the model based on the tuning parameters
	    	featObjs[i].score = objScore.mean();
	    	cout<<"****Finished object "<<i<<"\n";
	    	
	    }
	    
	    bestScoreIndex = 0;
	    
	    // finding the best model from validation and printing its parameters
	    for(int i = 1; i < featObjs.size(); i++)
	    	if(featObjs[i].score < featObjs[bestScoreIndex].score)
	    		bestScoreIndex = i;
	    		
	    cout << "\n********************\nBest tuning parameters for this data is \n";
	    cout << "\nML = " << featObjs[bestScoreIndex].obj.get_ml();
	    cout << "\nPopulation size = " << featObjs[bestScoreIndex].obj.get_pop_size();
	    cout << "\nGenerations = " << featObjs[bestScoreIndex].obj.get_generations();
	    cout << "\nDimensions = " << featObjs[bestScoreIndex].obj.get_max_dim();
	    cout << "\nDepth = " << featObjs[bestScoreIndex].obj.get_max_depth();
	    cout << "\nCross Rate = " <<featObjs[bestScoreIndex].obj.get_cross_rate();
	    cout << "\nScore = " << featObjs[bestScoreIndex].score<<"\n";
    }
    
    VectorXd FeatCV::predict(MatrixXd x)
    {
        // report error if data not trained first
    	if(bestScoreIndex == -1)
    	{
    		std::cerr << "You need to call fit first to cross validate first.\n";
    		throw;
    	}
    	
    	// prediciting values based on the best model found
    	return featObjs[bestScoreIndex].obj.predict(x);
    }    
}
#endif
