/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "data.h"
#include "../util/rnd.h"
#include "../util/logger.h"

//#include "node/node.h"
//external includes

namespace FT{

    using namespace Util;
    
    namespace Dat{

        Data::Data(MatrixXf& X, VectorXf& y, std::map<string, std::pair<vector<ArrayXf>, 
                        vector<ArrayXf>>>& Z, bool c): X(X), y(y), Z(Z), classification(c) 
        {
            validation=false;
        }
        
        void Data::set_validation(bool v){validation=v;}
        
        void Data::get_batch(Data &db, int batch_size) const
        {

            batch_size =  std::min(batch_size,int(y.size()));
            vector<size_t> idx(y.size());
            std::iota(idx.begin(), idx.end(), 0);
    //        r.shuffle(idx.begin(), idx.end());
            db.X.resize(X.rows(),batch_size);
            db.y.resize(batch_size);
            for (const auto& val: Z )
            {
                db.Z[val.first].first.resize(batch_size);
                db.Z[val.first].second.resize(batch_size);
            }
            for (unsigned i = 0; i<batch_size; ++i)
            {
               
               db.X.col(i) = X.col(idx.at(i)); 
               db.y(i) = y(idx.at(i)); 

               for (const auto& val: Z )
               {
                    db.Z[val.first].first.at(i) = Z.at(val.first).first.at(idx.at(i));
                    db.Z[val.first].second.at(i) = Z.at(val.first).second.at(idx.at(i));
               }
            }
        }
        
        DataRef::DataRef()
        {
            oCreated = false;
            tCreated = false;
            vCreated = false;
        }
     
        DataRef::DataRef(MatrixXf& X, VectorXf& y, 
                         std::map<string,std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z, bool c)
        {
            o = new Data(X, y, Z, c);
            oCreated = true;
            
            t = new Data(X_t, y_t, Z_t, c);
            tCreated = true;
            
            v = new Data(X_v, y_v, Z_v, c);
            vCreated = true;
            
            classification = c;
          
            // split data into training and test sets
            //train_test_split(params.shuffle, params.split);
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
        
        void DataRef::setOriginalData(MatrixXf& X, VectorXf& y, 
                                      std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z,
                                      bool c)
        {
            o = new Data(X, y, Z, c);
            oCreated = true;
            
            t = new Data(X_t, y_t, Z_t, c);
            tCreated = true;
            
            v = new Data(X_v, y_v, Z_v, c);
            vCreated = true;
            
            classification = c;
        }
        
        void DataRef::setOriginalData(Data *d)
        {
            o = d;
            oCreated = false;
            
            t = new Data(X_t, y_t, Z_t, d->classification);
            tCreated = true;
            
            v = new Data(X_v, y_v, Z_v, d->classification);
            vCreated = true;
            
            classification = d->classification;
        }
        
        void DataRef::setTrainingData(MatrixXf& X_t, VectorXf& y_t, 
                                    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z_t,
                                    bool c)
        {
            t = new Data(X_t, y_t, Z_t, c);
            tCreated = true;
            
            classification = c;
        }
        
        void DataRef::setTrainingData(Data *d, bool toDelete)
        {
            t = d;
            if(!toDelete)
                tCreated = false;
            else
                tCreated = true;
        }
        
        void DataRef::setValidationData(MatrixXf& X_v, VectorXf& y_v, 
                                    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>>& Z_v,
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
        
        void DataRef::shuffle_data()
        {
            Eigen::PermutationMatrix<Dynamic,Dynamic> perm(o->X.cols());
            perm.setIdentity();
            r.shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
            /* cout << "X before shuffle: \n"; */
            /* cout << o->X.transpose() << "\n"; */
            o->X = o->X * perm;       // shuffles columns of X

            /* cout << "X after shuffle: \n"; */
            /* cout << o->X.transpose() << "\n"; */
            o->y = (o->y.transpose() * perm).transpose() ;       // shuffle y too
            
            if(o->Z.size() > 0)
            {
                std::vector<int> zidx(o->y.size());
                // zidx maps the perm_indices values to their indices, i.e. the inverse transform
                for (unsigned i = 0; i < perm.indices().size(); ++i)
                    zidx.at(perm.indices()(i)) = i;
                /* cout << "zidx :\n"; */
                /* for (const auto& zi : zidx) */
                /*     cout << zi << "," ; */
                /* cout << "\n"; */
                for(auto &val : o->Z)
				{
                    /* cout << "unshuffled " << val.first << ": \n"; */
                    /* for (unsigned i = 0; i < val.second.first.size(); ++i) */
                    /* { */
                        /* cout << val.second.first.at(i).transpose() << "\n"; */
                    /* } */
                    reorder_longitudinal(val.second.first, zidx);
                    reorder_longitudinal(val.second.second, zidx);
                    /* cout << "shuffled " << val.first << ": \n"; */
                    /* for (unsigned i = 0; i < val.second.first.size(); ++i) */
                    /* { */
                    /*     cout << val.second.first.at(i).transpose() << "\n"; */
                    /* } */
				}
            }
        }
        
        void DataRef::split_stratified(float split)
        {
            logger.log("Stratify split called with initial data size as " + o->X.cols(), 3);
                            
            std::map<float, vector<int>> label_indices;
                
            //getting indices for all labels
            for(int x = 0; x < o->y.size(); x++)
                label_indices[o->y[x]].push_back(x);
                    
            std::map<float, vector<int>>::iterator it = label_indices.begin();
            
            vector<int> t_indices;
            vector<int> v_indices;
            
            int t_size;
            int x;
            
            for(; it != label_indices.end(); it++)
            {
                t_size = ceil(it->second.size()*split);
                
                for(x = 0; x < t_size; x++)
                    t_indices.push_back(it->second[x]);
                    
                for(; x < it->second.size(); x++)
                    v_indices.push_back(it->second[x]);
                
                logger.log("Label is " + to_string(it->first), 3, "\t");
                logger.log("Total size = " + it->second.size(), 3, "\t");
                logger.log("training_size = " + t_size, 3, "\t");
                logger.log("verification size = " + (it->second.size() - t_size), 3, "\t");
                
            }
            
            X_t.resize(o->X.rows(), t_indices.size());
            X_v.resize(o->X.rows(), v_indices.size());
            y_t.resize(t_indices.size());
            y_v.resize(v_indices.size());
            
            sort(t_indices.begin(), t_indices.end());
            
            for(int x = 0; x < t_indices.size(); x++)
            {
                t->X.col(x) = o->X.col(t_indices[x]);
                t->y[x] = o->y[t_indices[x]];
                
                if(o->Z.size() > 0)
                {
                    for(auto const &val : o->Z)
                    {
                        t->Z[val.first].first.push_back(val.second.first[t_indices[x]]);
                        t->Z[val.first].second.push_back(val.second.second[t_indices[x]]);
                    }
                }
            }
            
            sort(v_indices.begin(), v_indices.end());
            
            for(int x = 0; x < v_indices.size(); x++)
            {
                v->X.col(x) = o->X.col(v_indices[x]);
                v->y[x] = o->y[v_indices[x]];
                
                if(o->Z.size() > 0)
                {
                    for(auto const &val : o->Z)
                    {
                        v->Z[val.first].first.push_back(val.second.first[t_indices[x]]);
                        v->Z[val.first].second.push_back(val.second.second[t_indices[x]]);
                    }
                }
            }

            
        }
     
        void DataRef::train_test_split(bool shuffle, float split)
        {
            /* @param X: n_features x n_samples matrix of training data
             * @param y: n_samples vector of training labels
             * @param shuffle: whether or not to shuffle X and y
             * @param[out] X_t, X_v, y_t, y_v: training and validation matrices
             */
             
                                     
            if (shuffle)     // generate shuffle index for the split
                shuffle_data();
                
            if(classification)
                split_stratified(split);
            else
            {        
                // resize training and test sets
                X_t.resize(o->X.rows(),int(o->X.cols()*split));
                X_v.resize(o->X.rows(),int(o->X.cols()*(1-split)));
                y_t.resize(int(o->y.size()*split));
                y_v.resize(int(o->y.size()*(1-split)));
                
                // map training and test sets  
                t->X = MatrixXf::Map(o->X.data(),t->X.rows(),t->X.cols());
                v->X = MatrixXf::Map(o->X.data()+t->X.rows()*t->X.cols(),
                                           v->X.rows(),v->X.cols());

                t->y = VectorXf::Map(o->y.data(),t->y.size());
                v->y = VectorXf::Map(o->y.data()+t->y.size(),v->y.size());
                if(o->Z.size() > 0)
                    split_longitudinal(o->Z, t->Z, v->Z, split);
            }

        }  
        
        void DataRef::split_longitudinal(
                                std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z,
                                std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z_t,
                                std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z_v,
                                float split)
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
                vector<ArrayXf> _Z_t_v, _Z_t_t, _Z_v_v, _Z_v_t;
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
        
		void DataRef::reorder_longitudinal(vector<ArrayXf> &v, vector<int> const &order )  {   
			for ( int s = 1, d; s < order.size(); ++ s ) {
				for ( d = order[s]; d < s; d = order[d] ) ;
				if ( d == s ) while ( d = order[d], d != s ) swap( v[s], v[d] );
			}
		}
    }
}
