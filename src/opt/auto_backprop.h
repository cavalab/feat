#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "../pop/nodevector.h"
#include "../dat/state.h"
#include "../pop/op/n_Dx.h"
#include "../eval/metrics.h"
#include "../pop/individual.h"
#include "../model/ml.h"
#include "../init.h"
#include "../params.h"

#include <cmath>
#include <shogun/labels/Labels.h>

using shogun::CLabels;
using Eigen::MatrixXf;
using Eigen::VectorXf;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::cout;
/**
TODO
------------------------
Integrate vectorList
Integrate pointers?
TODO Make it so stops traversing once it hits a non-differentiable node and then goes upstream and finds another branch to traverse
**/

namespace FT {

    /**
     * @namespace FT::Opt
     * @brief namespace for backprop classes in Feat
     */
    namespace Opt{

        struct BP_NODE
	    {
		    NodeDx* n;
		    vector<ArrayXf> deriv_list;
	    };

       
	    template <class T>
	    T pop(vector<T>* v) {
		    T value = v->back();
		    v->pop_back();
		    return value;
	    }

	    template <class T>
	    T pop_front(vector<T>* v) {
		    T value = v->front();
		    v->erase(v->begin());
		    return value;
	    }
	
	    class AutoBackProp 
        {
            /* @class AutoBackProp
             * @brief performs back propagation on programs to adapt weights.
             */
	    public:
	
            typedef VectorXf (*callback)(const VectorXf&, shared_ptr<CLabels>&, const vector<float>&);
            
            std::map<string, callback> d_score_hash;
            std::map<string, callback> score_hash;
                    
            AutoBackProp(string scorer, int iters=1000, float n=0.1, float a=0.9); 

            /// adapt weights
		    void run(Individual& ind, const Data& d,
                     const Parameters& params);

            /* ~AutoBackProp() */
            /* { */
            /*     /1* for (const auto& p: program) *1/ */
            /*         /1* p = nullptr; *1/ */
            /* } */


	    private:
		    float n;                   //< learning rate
            float a;                   //< momentum
            callback d_cost_func;       //< derivative of cost function pointer
            callback cost_func;         //< cost function pointer
            
            int iters;                  //< iterations
            float epk;                 //< current learning rate 
            float epT;                  //< min learning rate

		    void print_weights(NodeVector& program);
		
		    /// Return the f_stack
		    vector<Trace> forward_prop(Individual& ind, const Data& d,
                                   MatrixXf& Phi, const Parameters& params);

		    /// Updates stacks to have proper value on top
		    void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                             vector<ArrayXf>& derivatives);

            /// Compute gradients and update weights 
            void backprop(Trace& f_stack, NodeVector& program, int start, int end, 
                                    float Beta, shared_ptr<CLabels>& yhat, 
                                    const Data& d,
                                   vector<float> sw);
                                   
            /// Compute gradients and update weights 
            void backprop2(Trace& f_stack, NodeVector& program, int start, int end, 
                                    float Beta, const VectorXf& yhat, 
                                    const Data& d,
                                   vector<float> sw);

	    };
	}
}

#endif
