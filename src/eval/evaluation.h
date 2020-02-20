/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H
// internal includes
#include "../model/ml.h"
#include "metrics.h"
#include "../opt/auto_backprop.h"
#include "../opt/hillclimb.h"
using namespace shogun;
using Eigen::Map;

// code to evaluate GP programs.
namespace FT{

    /**
     * @namespace FT::Eval
     * @brief namespace containing various Evaluation classes used in Feat
     */
    namespace Eval{
    
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Evaluation
         * @brief evaluation mixin class for Feat
         */
        typedef float (*funcPointer)(const VectorXf&, const shared_ptr<CLabels>&, VectorXf&,
                                      const vector<float>&);
        
        class Evaluation 
        {
            public:
            
                float (* score)(const VectorXf&, const shared_ptr<CLabels>&, VectorXf&, 
                                 const vector<float>&);    // pointer to scoring function
                                 
                std::map<string, funcPointer> score_hash;

                Evaluation(string scorer);

                ~Evaluation();
                    
                void set_score(string scorer);

                /// validation of population.
                void validation(vector<Individual>& individuals,
                             const Data& d, 
                             const Parameters& params, 
                             bool offspring = false
                             );

                /// fitness of population.
                void fitness(vector<Individual>& individuals,
                             const Data& d, 
                             MatrixXf& F, 
                             const Parameters& params, 
                             bool offspring = false
                             );
              
                /// assign fitness to an individual and to F.  
                void assign_fit(Individual& ind, MatrixXf& F, const shared_ptr<CLabels>& yhat, 
                                const VectorXf& y, const Parameters& params,bool val=false);       
        };
    }
}
#endif
