/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "hillclimb.h"
#include "../eval/metrics.h"
#include "../ml/ml.h"

namespace FT {
        
    HillClimb::HillClimb(string scorer, int iters, double step)
    {
        score_hash["mse"] = & metrics::squared_difference;
        score_hash["log"] =  & metrics::log_loss; 
        score_hash["multi_log"] =  & metrics::multi_log_loss;
        this->cost_func = score_hash[scorer]; 
		/* this->X = X; */
		/* this->labels = labels; */
		this->iters = iters;
		this->step = step;

    }
    
    shared_ptr<CLabels> HillClimb::run(Individual& ind, Data d,
                 const Parameters& params, bool& updated)
    {
        updated = false;    // keep track of whether we update this individual
        double min_loss = ind.fitness; 
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
                    vector<double> W0 = pd->W;
                    for (int i = 0; i < pd->W.size(); ++i)
                    {   // perturb W[i] with gaussian noise
                        pd->W[i] += r.gasdev()*pd->W[i]*this->step; 
                        /* cout << "changed " << W0[i] << " to " << pd->W[i] << "\n"; */
                    }
                }
            }
            if (!anychanges)    // then there are no weighted nodes, so break
                break;
            // evaluate perturbed program 
            params.msg("Generating output for " + tmp.get_eqn(), 3);
            MatrixXd Phi = tmp.out(d, params);            

            // calculate ML model from Phi
            params.msg("ML training on " + tmp.get_eqn(), 3);
            bool pass = true;
            auto ml = std::make_shared<ML>(params);

            shared_ptr<CLabels> yhat = ml->fit(Phi,d.y,params,pass,tmp.dtypes);
            
            if (!pass)
                continue;

            double new_loss = this->cost_func(d.y,yhat, params.class_weights).mean();
            /* cout << "old loss: " << min_loss << ", new_loss: " << new_loss << "\n"; */
            if (new_loss < min_loss)
            {
                updated = true;
                /* cout << "updating \n"; */
                ind.program = tmp.program;
                ind.fitness = new_loss; 
                min_loss = new_loss;
                ind.set_p(ml->get_weights(),params.feedback);
                min_yhat = yhat;
            }
        }
        return min_yhat;
    }

}
