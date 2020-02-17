/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "hillclimb.h"
#include "../eval/metrics.h"
#include "../model/ml.h"

namespace FT {

    namespace Opt{
        
        HillClimb::HillClimb(string scorer, int iters, float step)
        {
            score_hash["mse"] = &Eval::squared_difference;
            score_hash["log"] =  &Eval::log_loss; 
            score_hash["multi_log"] =  &Eval::multi_log_loss;
            this->cost_func = score_hash.at(scorer); 
		    /* this->X = X; */
		    /* this->labels = labels; */
		    this->iters = iters;
		    this->step = step;

        }
        
        shared_ptr<CLabels> HillClimb::run(Individual& ind, Data d,
                     const Parameters& params, bool& updated)
        {
            updated = false;    // keep track of whether we update this individual
            float min_loss = ind.fitness; 
            shared_ptr<CLabels> min_yhat;

            for (int x = 0; x < this->iters; x++)
            {
                /* cout << "iteration " << x << "\n"; */
                Individual tmp;
                ind.clone(tmp);
                bool anychanges = false;
                for (auto& p : tmp.program)
                {
                    if (p->isNodeDx())
                    {
                        anychanges = true;
                        auto pd = dynamic_cast<NodeDx*>(p.get());
                        vector<float> W0 = pd->W;
                        for (int i = 0; i < pd->W.size(); ++i)
                        {   // perturb W[i] with gaussian noise
                            pd->W.at(i) += r.gasdev()*pd->W.at(i)*this->step; 
                            /* cout << "changed " << W0.at(i) << " to " << pd->W.at(i) << "\n"; */
                        }
                    }
                }
                if (!anychanges)    // then there are no weighted nodes, so break
                    break;
                // evaluate perturbed program 
                logger.log("Generating output for " + tmp.get_eqn(), 3);

                bool pass = true;

                shared_ptr<CLabels> yhat = tmp.fit(d, params, pass);
                if (!pass)
                    continue;

                float new_loss = this->cost_func(d.y,yhat, params.class_weights).mean();
                /* cout << "old loss: " << min_loss << ", new_loss: " << new_loss << "\n"; */
                if (new_loss < min_loss)
                {
                    updated = true;
                    /* cout << "updating \n"; */
                    ind.program = tmp.program;
                    ind.fitness = new_loss; 
                    min_loss = new_loss;
                    /* ind.set_p(ind.ml->get_weights(),params.feedback); */
                    min_yhat = yhat;
                }
            }
            return min_yhat;
        }
    }

}
