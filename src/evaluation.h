/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H
// internal includes
#include "ml.h"
#include "metrics.h"
#include "auto_backprop.h"
#include "hillclimb.h"
using namespace shogun;
using Eigen::Map;

// code to evaluate GP programs.
namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Evaluation
     * @brief evaluation mixin class for Feat
     */
    typedef double (*funcPointer)(const VectorXd&, const shared_ptr<CLabels>&, VectorXd&,
                                  const vector<float>&);
    
    class Evaluation 
    {
        public:
        
            double (* score)(const VectorXd&, const shared_ptr<CLabels>&, VectorXd&, 
                             const vector<float>&);    // pointer to scoring function
                             
            std::map<string, funcPointer> score_hash;

            Evaluation(string scorer);

            ~Evaluation();
                
            void set_score(string scorer);

            /// fitness of population.
            void fitness(vector<Individual>& individuals,
                         const Data& d, 
                         MatrixXd& F, 
                         const Parameters& params, 
                         bool offspring = false,
                         bool validation = false);
          
            /// assign fitness to an individual and to F.  
            void assign_fit(Individual& ind, MatrixXd& F, const shared_ptr<CLabels>& yhat, 
                            const VectorXd& y, const Parameters& params,bool val=false);       
    };
}
#endif
