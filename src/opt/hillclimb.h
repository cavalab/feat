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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <shogun/labels/Labels.h>
#pragma GCC diagnostic pop

#include <string>

#include "../pop/individual.h"
#include "../dat/data.h"
#include "../params.h"

using std::map;
using std::shared_ptr;
using std::vector;
using Eigen::VectorXf;
using shogun::CLabels;

namespace FT {

    namespace Opt{
        class HillClimb
        {
            /* @class HillClimb
             * @brief performs random weight updates and keeps them if they improve the cost function.
             */
        public:
            typedef VectorXf (*callback)(const VectorXf&, shared_ptr<CLabels>&, const vector<float>&);
            
            std::map<string, callback> score_hash;
            
            HillClimb(string scorer, int iters=1, float step=0.1);

            /// adapt weights
		    shared_ptr<CLabels> run(Individual& ind, Data d,
                     const Parameters& params, bool& updated);

        private:
            callback cost_func;     //< scoring function
            int iters;              //< number of iterations
            float step;            //< percent of std dev to perturb weight by
        };
    }

}
#endif
