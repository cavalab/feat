/* FEAT
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
    
    bool string2bool (const std::string & v);
    
    tokenType getTokenType(string &token);
    
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
}
#endif
