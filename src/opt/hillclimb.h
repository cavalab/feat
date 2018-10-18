/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef HILLCLIMB_H
#define HILLCLIMB_H

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <vector>
#include <shogun/labels/Labels.h>
#include <string>

#include "../pop/individual.h"
#include "../data/data.h"
#include "../params.h"

using std::map;
using std::shared_ptr;
using std::vector;
using Eigen::VectorXd;
using shogun::CLabels;

namespace FT {

    namespace Opt{
        class HillClimb
        {
            /* @class HillClimb
             * @brief performs random weight updates and keeps them if they improve the cost function.
             */
        public:
            typedef VectorXd (*callback)(const VectorXd&, shared_ptr<CLabels>&, const vector<float>&);
            
            std::map<string, callback> score_hash;
            
            HillClimb(string scorer, int iters=1, double step=0.1);

            /// adapt weights
		    shared_ptr<CLabels> run(Individual& ind, Data d,
                     const Parameters& params, bool& updated);

        private:
            callback cost_func;     //< scoring function
            int iters;              //< number of iterations
            double step;            //< percent of std dev to perturb weight by
        };
    }

}
#endif
