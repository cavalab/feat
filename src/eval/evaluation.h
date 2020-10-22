/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H
// internal includes
#include "../model/ml.h"
#include "scorer.h"
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
    
        ////////////////////////////////////////////////////////// Declarations
        /*!
         * @class Evaluation
         * @brief evaluation mixin class for Feat
         */
        /* typedef float (*funcPointer)(const VectorXf&, */ 
        /*                              const shared_ptr<CLabels>&, VectorXf&, */
        /*                               const vector<float>&); */
        
        class Evaluation 
        {
            public:
            
                /* float (* score)(const VectorXf&, */ 
                /*                 const shared_ptr<CLabels>&, */ 
                /*                 VectorXf&, */ 
                /*                 const vector<float>&);    // pointer to scoring function */
                                 
                /* std::map<string, funcPointer> score_hash; */

                Evaluation(string scorer="");

                ~Evaluation();
                    
                /* void set_score(string scorer); */

                /// validation of population.
                void validation(vector<Individual>& individuals,
                             const Data& d, 
                             const Parameters& params, 
                             bool offspring = false
                             );

                /// fitness of population.
                void fitness(vector<Individual>& individuals,
                             const Data& d, 
                             const Parameters& params, 
                             bool offspring = false
                             );
              
                 
                float marginal_fairness(VectorXf& loss, const Data& d, 
                        float base_score, bool use_alpha=false);

                /// assign fitness to an individual.  
                void assign_fit(Individual& ind, 
                        const shared_ptr<CLabels>& yhat, 
                        const Data& d, 
                        const Parameters& params,bool val=false);       

                Scorer S;
        };
    }
}
#endif
