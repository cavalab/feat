/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef DATA_H
#define DATA_H

#include <string>
#include <Eigen/Dense>
#include <vector>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::VectorXi;
using Eigen::Dynamic;
using Eigen::Map;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using namespace std;
// internal includes
//#include "params.h"
#include "utils.h"
//#include "node/node.h"
//external includes

namespace FT
{
    /*!
     * @class Data
     * @brief data holding X, y, and Z data
     */
    class Data
    {
        //Data(MatrixXd& X, VectorXd& y, std::map<string, 
        //std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z): X(X), y(y), Z(Z){}
        public:
 
            MatrixXd& X;
            VectorXd& y;
            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z;
            bool classification;
            bool validation; 

            Data(MatrixXd& X, VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, 
                    vector<ArrayXd>>>& Z, bool c = false): X(X), y(y), Z(Z), classification(c) 
            {
                validation=false;
            }

            void set_validation(bool v=true){validation=v;}
    };
    
    class DataRef
    {
        private:
            bool oCreated;
            bool tCreated;
            bool vCreated;
            // training and validation data
            MatrixXd X_t;
            MatrixXd X_v;
            VectorXd y_t;
            VectorXd y_v;
            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_t;
            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z_v;

        public:
            Data *o = NULL;     //< pointer to original data
            Data *v = NULL;     //< pointer to validation data
            Data *t = NULL;     //< pointer to training data
            
            DataRef();
            
            ~DataRef();
            
    
            DataRef(MatrixXd& X, VectorXd& y, 
                             std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z, 
                             bool c=false);
            /* DataRef(MatrixXd& X, VectorXd& y, std::map<string, */ 
            /*         std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z, */
            /*         MatrixXd& X_t, VectorXd& y_t, std::map<string, */ 
            /*         std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t, */
            /*         MatrixXd& X_v, VectorXd& y_v, std::map<string, std::pair<vector<ArrayXd>, */ 
            /*         vector<ArrayXd>>>& Z_v); */
                    
            void setOriginalData(MatrixXd& X, VectorXd& y, 
                    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z, bool c=false);
            
            void setOriginalData(Data *d);
            
            void setTrainingData(MatrixXd& X_t, VectorXd& y_t, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t,
                               bool c = false);
            
            void setTrainingData(Data *d);
            
            void setValidationData(MatrixXd& X_v, VectorXd& y_v, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v,
                               bool c = false);
            
            void setValidationData(Data *d);
           
            /// splits data into training and validation folds.
            void train_test_split(bool shuffle, double split);

            /// reorders longitudinal data according to shuffle
            /* void reorder_longitudinal(vector<ArrayXd> &vec1, */
            /*                  vector<ArrayXd> &vec2, */
            /*                  vector<int> const &order); */ 

            void split_longitudinal(
                        std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                        std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                        std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                        double split);

    };
    
    DataRef::DataRef()
    {
        oCreated = false;
        tCreated = false;
        vCreated = false;
    }
 
    DataRef::DataRef(MatrixXd& X, VectorXd& y, 
                     std::map<string,std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z, bool c)
    /* DataRef::DataRef(MatrixXd& X, VectorXd& y, */ 
    /*                  std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z, */
    /*                  MatrixXd& X_t, VectorXd& y_t, */ 
    /*                  std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t, */
    /*                  MatrixXd& X_v, VectorXd& y_v, */ 
    /*                  std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v) */
    {
        o = new Data(X, y, Z, c);
        oCreated = true;
        
        t = new Data(X_t, y_t, Z_t, c);
        tCreated = true;
        
        v = new Data(X_v, y_v, Z_v, c);
        vCreated = true;
      
        // split data into training and test sets
        /* train_test_split(params.shuffle, params.split); */
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
    
    void DataRef::setOriginalData(MatrixXd& X, VectorXd& y, 
                                  std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z,
                                  bool c)
    {
        o = new Data(X, y, Z, c);
        oCreated = true;
    }
    
    void DataRef::setOriginalData(Data *d)
    {
        o = d;
        oCreated = false;
    }
    
    void DataRef::setTrainingData(MatrixXd& X_t, VectorXd& y_t, 
                                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_t,
                                bool c)
    {
        t = new Data(X_t, y_t, Z_t, c);
        tCreated = true;
    }
    
    void DataRef::setTrainingData(Data *d)
    {
        t = d;
        tCreated = false;
    }
    
    void DataRef::setValidationData(MatrixXd& X_v, VectorXd& y_v, 
                                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>>& Z_v,
                                bool c)
    {
        v = new Data(X_v, y_v, Z_v, c);
        vCreated = true;
    }
    
    void DataRef::setValidationData(Data *d)
    {
        v = d;
        vCreated = false;
    }
 
    void DataRef::train_test_split(bool shuffle, double split)
    {
        /* @params X: n_features x n_samples matrix of training data
         * @params y: n_samples vector of training labels
         * @params shuffle: whether or not to shuffle X and y
         * @returns X_t, X_v, y_t, y_v: training and validation matrices
         */
        
        // resize training and test sets
        X_t.resize(o->X.rows(),int(o->X.cols()*split));
        X_v.resize(o->X.rows(),int(o->X.cols()*(1-split)));
        y_t.resize(int(o->y.size()*split));
        y_v.resize(int(o->y.size()*(1-split)));
 
        if (shuffle)     // generate shuffle index for the split
        {
            Eigen::PermutationMatrix<Dynamic,Dynamic> perm(o->X.cols());
            perm.setIdentity();
            r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
            o->X = o->X * perm;       // shuffles columns of X
            o->y = (o->y.transpose() * perm).transpose() ;       // shuffle y too
            
            if(o->Z.size() > 0)
            {
                std::vector<int> zidx(o->y.size());
                std::iota(zidx.begin(), zidx.end(), 0);
                Eigen::VectorXi zw = Map<VectorXi>(zidx.data(), zidx.size());
                zw = (zw.transpose()*perm).transpose();       // shuffle zw too
                zidx.assign(((int*)zw.data()), (((int*)zw.data())+zw.size()));
                
                for(auto &val : o->Z)
                    reorder_longitudinal(val.second.first, val.second.second, zidx);
            }
            
        }
        
        // map training and test sets  
        t->X = MatrixXd::Map(o->X.data(),t->X.rows(),t->X.cols());
        v->X = MatrixXd::Map(o->X.data()+t->X.rows()*t->X.cols(),
                                   v->X.rows(),v->X.cols());

        t->y = VectorXd::Map(o->y.data(),t->y.size());
        v->y = VectorXd::Map(o->y.data()+t->y.size(),v->y.size());
        
        if(o->Z.size() > 0)
            split_longitudinal(o->Z, t->Z, v->Z, split);

    }  
    
    /* void DataRef::reorder_longitudinal(vector<ArrayXd> &vec1, */
    /*                          vector<ArrayXd> &vec2, */
    /*                          vector<int> const &order) */ 
    /* { */   
    
    /*     for( int s = 1, d; s < order.size(); ++ s ) */
    /*     { */
    /*         for ( d = order[s]; d < s; d = order[d] ); */
            
    /*         if ( d == s ) */
    /*         { */
    /*             while ( d = order[d], d != s ) */
    /*             { */
    /*                 swap(vec1[s], vec1[d]); */
    /*                 swap(vec2[s], vec2[d]); */
    /*             } */
    /*         } */
    /*     } */
    /* } */
    
    void DataRef::split_longitudinal(
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_t,
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z_v,
                            double split)
    {
    
        int size;
        for ( const auto val: Z )
        {
            size = Z[val.first].first.size();
            break;
        }
        
        int testSize = int(size*split);
        int validateSize = int(size*(1-split));
            
        for ( const auto &val: Z )
        {
            vector<ArrayXd> _Z_t_v, _Z_t_t, _Z_v_v, _Z_v_t;
            _Z_t_v.assign(Z[val.first].first.begin(), Z[val.first].first.begin()+testSize);
            _Z_t_t.assign(Z[val.first].second.begin(), Z[val.first].second.begin()+testSize);
            _Z_v_v.assign(Z[val.first].first.begin()+testSize, 
                          Z[val.first].first.begin()+testSize+validateSize);
            _Z_v_t.assign(Z[val.first].second.begin()+testSize, 
                          Z[val.first].second.begin()+testSize+validateSize);
            
            Z_t[val.first] = make_pair(_Z_t_v, _Z_t_t);
            Z_v[val.first] = make_pair(_Z_v_v, _Z_v_t);
        }
    }    
}



#endif
