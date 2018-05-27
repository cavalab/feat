/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "data.h"

//#include "node/node.h"
//external includes

namespace FT
{    
    DataRef::DataRef()
    {
        oCreated = false;
        tCreated = false;
        vCreated = false;
    }
    
    DataRef::DataRef(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z,
                     MatrixXd& X_t, VectorXd& y_t, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t,
                     MatrixXd& X_v, VectorXd& y_v, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v)
    {
        o = new Data(X, y, Z);
        oCreated = true;
        
        t = new Data(X_t, y_t, Z_t);
        tCreated = true;
        
        v = new Data(X_v, y_v, Z_v);
        vCreated = true;
    }
   
    DataRef::~DataRef()
    {
        if(o != NULL && oCreated)
        {
            delete(o);
            o = NULL;
        }
        
        if(t != NULL && tCreated)
        {
            delete(t);
            t = NULL;
        }
        
        if(v != NULL && vCreated)
        {
            delete(v);
            v = NULL;
        }
    }
    
    void DataRef::setOriginalData(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z)
    {
        o = new Data(X, y, Z);
        oCreated = true;
    }
    
    void DataRef::setOriginalData(Data *d)
    {
        o = d;
        oCreated = false;
    }
    
    void DataRef::setTrainingData(MatrixXd& X_t, VectorXd& y_t, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t)
    {
        t = new Data(X_t, y_t, Z_t);
        tCreated = true;
    }
    
    void DataRef::setTrainingData(Data *d)
    {
        t = d;
        tCreated = false;
    }
    
    void DataRef::setValidationData(MatrixXd& X_v, VectorXd& y_v, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v)
    {
        v = new Data(X_v, y_v, Z_v);
        vCreated = true;
    }
    
    void DataRef::setValidationData(Data *d)
    {
        v = d;
        vCreated = false;
    }
    
}
