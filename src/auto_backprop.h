#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "nodevector.h"
#include "stack.h"
#include "node/node.h"
#include "node/n_Dx.h"
#include "metrics.h"
#include "individual.h"
#include "ml.h"
#include "init.h"
#include "params.h"

#include <cmath>
#include <shogun/labels/Labels.h>

using shogun::CLabels;
using Eigen::MatrixXd;
using Eigen::VectorXd;
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
	class AutoBackProp 
    {
        /* @class AutoBackProp
         * @brief performs back propagation on programs to adapt weights.
         */
	public:
	
        typedef VectorXd (*callback)(const VectorXd&, shared_ptr<CLabels>&, const vector<float>&);
        
        std::map<string, callback> d_score_hash;
        std::map<string, callback> score_hash;
        
        AutoBackProp(string scorer, int iters=1000, double n=0.1, double a=0.9); 

        /// adapt weights
		void run(Individual& ind, Data d,
                 const Parameters& params);

        /* ~AutoBackProp() */
        /* { */
        /*     /1* for (const auto& p: program) *1/ */
        /*         /1* p = nullptr; *1/ */
        /* } */


	private:
		double n;                   //< learning rate
        double a;                   //< momentum
        callback d_cost_func;       //< derivative of cost function pointer
        callback cost_func;         //< cost function pointer
        int iters;                  //< iterations
        double epk;                 //< current learning rate 
        double epT;                  //< min learning rate

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		void print_weights(NodeVector& program);
		
		/// Return the f_stack
		vector<Trace> forward_prop(Individual& ind, Data d,
                               MatrixXd& Phi, const Parameters& params);

		/// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                         vector<ArrayXd>& derivatives);

        /// Compute gradients and update weights 
        void backprop(Trace& f_stack, NodeVector& program, int start, int end, 
                                double Beta, shared_ptr<CLabels>& yhat, 
                                Data d,
                               vector<float> sw);

        /// select random subset of data for training weights.
        void get_batch(Data d, Data db, int batch_size);
       
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


	};
}

#endif
