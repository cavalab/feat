/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
// internal includes
#include "metrics.h"
#include "scorer.h"
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

            Scorer::Scorer(string scorer)
            {
                cout << "setting score_hash\n";
                score_hash["mse"] = &mse_label;
                score_hash["zero_one"] = &zero_one_loss_label;
                score_hash["bal_zero_one"] = &bal_zero_one_loss_label;
                score_hash["log"] =  &log_loss_label; 
                score_hash["multi_log"] =  &multi_log_loss_label; 
                score_hash["fpr"] =  &false_positive_loss_label; 
            
                cout << "setting scorer to " << scorer << "\n";
                this->set_scorer(scorer);
            }
            void Scorer::set_scorer(string scorer)
            {
                this->scorer = scorer;
                /* score = score_hash.at(scorer); */
                cout << "this->scorer set to : " << this->scorer << endl;
            }

            float Scorer::score(const VectorXf& y_true, 
                               const shared_ptr<CLabels>& yhat,
                               VectorXf& loss, 
                               const vector<float>& w)
            {
                cout << "this->scorer: " << this->scorer << endl;
                if ( score_hash.find(this->scorer) == score_hash.end() ) 
                {
                    // not found
                    HANDLE_ERROR_THROW("Scoring function '" + this->scorer
                            + "' not defined");
                } 
                else 
                {
                    // found
                    return score_hash.at(this->scorer)(y_true, yhat, loss, w); 
                }
            }
            // overloaded score with no loss
            float Scorer::score(const VectorXf& y_true, 
                               const shared_ptr<CLabels>& yhat,
                               vector<float> w)
            {
                VectorXf dummy;
                return this->score(y_true, yhat, dummy, w);
            }
    }
}
