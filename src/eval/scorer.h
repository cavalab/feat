/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef SCORER_H
#define SCORER_H
// internal includes
#include "metrics.h"
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
         * @class Scorer
         * @brief scoring class for Feat
         */
        typedef float (*funcPointer)(const VectorXf&, 
                                     const shared_ptr<CLabels>&, 
                                     VectorXf&,
                                     const vector<float>&);

        class Scorer
        {
            public:
                /* // pointer to scoring function */
                /* float (* score_fn)(const VectorXf&, */ 
                /*                 const shared_ptr<CLabels>&, */ 
                /*                 VectorXf&, */ 
                /*                 const vector<float>&); */    

                std::map<string, funcPointer> score_hash;

                Scorer(string scorer="");

                void set_scorer(string scorer);
                /* void set_scorer(string scorer); */
                float score(const VectorXf& y_true, 
                               const shared_ptr<CLabels>& yhat,
                               VectorXf& loss, 
                               const vector<float>& w);
                float score(const VectorXf& y_true, 
                               const shared_ptr<CLabels>& yhat,
                               vector<float> w=vector<float>());

                string scorer;
        };
    }
}
#endif
