#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "nodevector.h"
#include "stack.h"
#include "node/node.h"
#include "node/nodeDx.h"
//#include "metrics.h"
#include "individual.h"

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
        
        AutoBackProp(string scorer, int iters=1000, double n=0.1) 
        {
			/* this->program = program.get_data(); */
            score_hash["mse"] = & metrics::squared_difference;
            score_hash["log"] =  & metrics::log_loss; 
            score_hash["multi_log"] =  & metrics::multi_log_loss;
	        d_score_hash["mse"] = & metrics::d_squared_difference;
            d_score_hash["log"] =  & metrics::d_log_loss; 
            d_score_hash["multi_log"] =  & metrics::d_multi_log_loss;
			
            this->d_cost_func = d_score_hash[scorer]; 
            this->cost_func = score_hash[scorer]; 
			/* this->X = X; */
			/* this->labels = labels; */
			this->iters = iters;
			this->n = n;
		}
        /// adapt weights
		void run(Individual& ind, const MatrixXd& X, VectorXd& y,
                 const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                 const Parameters& params);

        /* ~AutoBackProp() */
        /* { */
        /*     /1* for (const auto& p: program) *1/ */
        /*         /1* p = nullptr; *1/ */
        /* } */


	private:
		double n;                   //< learning rate
		callback d_cost_func;       //< derivative of cost function pointer
        callback cost_func;         //< cost function pointer
        int iters;                  //< iterations

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		void print_weights(NodeVector& program) {
			for (const auto& p : program) 
            {
				cout << "( " << p->name;
				if (isNodeDx(p)) {
                    
					NodeDx* dNode = dynamic_cast<NodeDx*>(p.get());
					for (int i = 0; i < dNode->arity['f']; i++) {
						cout << "," << dNode->W.at(i);
					}
                    dNode = nullptr;
				}

				cout << " ) ";
			}
            /* cout << "\n"; */
		}
		/// Return the f_stack
		vector<vector<ArrayXd>> forward_prop(Individual& ind, const MatrixXd& X, VectorXd& y, 
                               const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                               MatrixXd& Phi, const Parameters& params);

		/// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                         vector<ArrayXd>& derivatives);

        /// Compute gradients and update weights 
		/* void backprop(vector<ArrayXd>& f_stack, NodeVector& program, int start, int end, */
                      /* MatrixXd& X, VectorXd& y, */ 
                               /* std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z); */
        void backprop(vector<ArrayXd>& f_stack, NodeVector& program, int start, int end, 
                                VectorXd& phi, double Beta, shared_ptr<CLabels>& yhat, 
                                const MatrixXd& X, VectorXd& y, 
                               const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                               vector<float> sw);

        
        bool isNodeDx(Node* n){ return NULL != dynamic_cast<NodeDx*>(n); ; }
        bool isNodeDx(const std::unique_ptr<Node>& n) 
        {
            Node * tmp = n.get();
            bool answer = isNodeDx(tmp); 
            tmp = nullptr;
            return answer;
        }
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

/////////////////////////////////////////////////////////////////////////////////////// Definitions
    // adapt weights 
    /* void AutoBackProp::run(NodeVector& program, MatrixXd& X, VectorXd& y, */ 
    /*                         std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z) */
    /* { */
    /*     cout << "Starting up AutoBackProp with " << this->iters << " iterations."; */
    /*     // Computes weights via backprop */
    /*     // grab subtrees to backprop over */
    /*     for (int s : program.roots()) */
    /*     { */
    /*         if (isNodeDx(program[s])) */
    /*         { */
    /*             cout << "\ntraining sub-program " << program.subtree(s) << " to " << s << "\n"; */
    /*             cout << "\nIteration\tLoss\tGrad\t\n"; */
    /*             for (int x = 0; x < this->iters; x++) { */
    /*                 // Evaluate forward pass */
    /*                 vector<ArrayXd> stack_f = forward_prop(program, program.subtree(s), s, X, y, Z); */
    /*                 // if ((x % round(this->iters/4)) == 0 or x == this->iters - 1) { */
    /*                 // } */
    /*                 cout << x << "\t" */ 
    /*                      << (y.array()-stack_f[stack_f.size() - 1]).array().pow(2).mean() << "\t" */
    /*                      << this->d_cost_func(y, stack_f[stack_f.size() - 1]).mean() << "\n"; */ 
                   
    /*                 // TODO: add ML output and weight/normalization of subtree to stack */
    /*                 // Evaluate backward pass */
    /*                 backprop(stack_f, program, program.subtree(s), s, X, y, Z); */
    /*             } */
    /*         } */
    /*     } */
    /*     cout << "Finished backprop. returning program:\n"; */
    /*     print_weights(program); */    
    /*     /1* return this->program; *1/ */
    /* } */
 
    void AutoBackProp::run(Individual& ind, const MatrixXd& X, VectorXd& y, 
                            const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                            const Parameters& params)
    {
        vector<size_t> roots = ind.program.roots();

        cout << "running backprop on " << ind.get_eqn() << "\n";
        cout << "params.sample_weights: " << params.sample_weights.size() << "\n";
        cout << "params.scorer: " << params.scorer << "\n";       
        cout << "=========================\n";
        cout << "Iteration,Loss,Weights\n";
        cout << "=========================\n";
        for (int x = 0; x < this->iters; x++)
        {
            // Evaluate forward pass
            MatrixXd Phi; 
            vector<vector<ArrayXd>> stack_f = forward_prop(ind, X, y, Z, Phi, params);
       
            /* cout << "stack_f size: " << stack_f.size() << "\n"; */
            /* int i = 0; */
            /* for (auto sf : stack_f) */
            /* { */
            /*     cout << "stack_f[" << i << "].size(): " << sf.size() << "\n"; */
            /*     ++i; */
            /* } */ 
            // Evaluate ML model on Phi
            bool pass = true;
            auto ml = std::make_shared<ML>(params, false);

            shared_ptr<CLabels> yhat = ml->fit(Phi,y,params,pass,ind.dtypes);

            if (!pass)
                continue;

            vector<double> Beta = ml->get_weights();

            cout << x << "," 
                 << this->cost_func(y,yhat, params.sample_weights).sum() << ",";
                  print_weights(ind.program);
            cout << "\n";
                 /* << this->d_cost_func(y, yhat, params.sample_weights).std() << "\n"; */ 
           
            // TODO: add ML output and weight/normalization of subtree to stack
            // Evaluate backward pass
            size_t s = 0;
            for (int i = 0; i < stack_f.size(); ++i)
            {
                while (!isNodeDx(ind.program.at(roots[s]))) ++s;
                /* cout << "running backprop on " << ind.program_str() << " from " << roots.at(s) << " to " */ 
                /*     << ind.program.subtree(roots.at(s)) << "\n"; */
                VectorXd phi = Phi.row(i);
                backprop(stack_f.at(i), ind.program, ind.program.subtree(roots.at(s)), roots.at(s),  
                         phi, Beta.at(i), yhat,
                         X, y, Z, params.sample_weights);
            }
        }
        cout << "=========================\n";
        cout << "done=====================\n";
        cout << "=========================\n";
    }
    /* // Return the f_stack */
    /* vector<ArrayXd> AutoBackProp::forward_prop(NodeVector& program, int start, int end, */ 
    /*                                             MatrixXd& X, VectorXd& y, */ 
    /*                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z) */ 
    /* { */
    /*     /1* cout << "Forward pass\n"; *1/ */
    /*     // Iterate through all the nodes evaluating and tracking ouputs */
    /*     vector<ArrayXd> stack_f; // Tracks output values */
    /*     vector<ArrayXd> execution_stack; // Tracks output values and groups them based on input to functions */
    /*     vector<ArrayXb> tmp; */

    /*     FT::Stacks stack; */

    /*     // Use stack_f and execution stack to avoid issue of branches affecting what elements */ 
    /*     // appear before a node */ 
    /*     /1* for (const auto& p : program) *1/ */ 
    /*     for (int s = start; s <= end; ++s) */ 
    /*     { // Can think about changing with call to Node for cases with nodes that aren't differentiable */
    /*         /1* cout << "Evaluating node: " << program[s]->name << "\n"; *1/ */
    /*         for (int i = 0; i < program[s]->arity['f']; i++) { */
    /*             stack_f.push_back(stack.f.at(stack.f.size() - (program[s]->arity['f'] - i))); */
    /*             // stack_f.push_back(execution_stack[execution_stack.size() - (p->arity['f'] - i)]); */
    /*         } */

    /*         program[s]->evaluate(X, y, Z, stack); // execution_stack, tmp); */
    /*         program[s]->visits = 0; */
    /*     } */

    /*     stack_f.push_back(stack.f.pop()); */

    /*     /1* cout << "Returning forward pass.\n"; *1/ */
    /*     return stack_f; */
    /* } */
 
    vector<vector<ArrayXd>> AutoBackProp::forward_prop(Individual& ind, const MatrixXd& X, VectorXd& y, 
                               const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                               MatrixXd& Phi, const Parameters& params) 
    {
        /* cout << "Forward pass\n"; */
        // Iterate through all the nodes evaluating and tracking ouputs
        vector<vector<ArrayXd>> stack_f_trace;
        Phi = ind.out_trace(X, Z, params, stack_f_trace, y);
        // Use stack_f and execution stack to avoid issue of branches affecting what elements 
        // appear before a node 
        /* cout << "Returning forward pass.\n"; */
        return stack_f_trace;
    }   
    // Updates stacks to have proper value on top
    void AutoBackProp::next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                                   vector<ArrayXd>& derivatives) 
    {
        // While there are still nodes with branches to explore
        if(!executing.empty()) {
            // Declare variable to hold node and its associated derivatives
            BP_NODE bp_node = pop<BP_NODE>(&executing); // Check first element
            // Loop until branch to explore is found
            while (bp_node.deriv_list.empty() && !executing.empty()) {
                bp_node = pop<BP_NODE>(&executing); // Get node and its derivatves

                // For some reason this function is not removing element from the stack
                pop<ArrayXd>(&derivatives); // Remove associated gradients from stack
                if (executing.empty()) {
                    return;
                }
            }
            
            // Should now have the next parent node and derivatves (stored in bp_node)
            if (!bp_node.deriv_list.empty()) 
            {
                bp_program.push_back(bp_node.n);
                // Pull derivative from front of list due to how we stored them earlier
                derivatives.push_back(pop_front<ArrayXd>(&(bp_node.deriv_list)));                 
                // Push it back on the stack in order to sync all the stacks
                executing.push_back(bp_node);             
            }
        }
    }

    // Compute gradients and update weights 
    void AutoBackProp::backprop(vector<ArrayXd>& f_stack, NodeVector& program, int start, int end, 
                                VectorXd& phi, double Beta, shared_ptr<CLabels>& yhat, 
                                const MatrixXd& X, VectorXd& y, 
                                const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z,
                                vector<float> sw)    
    {
        /* cout << "Backward pass \n"; */
        vector<ArrayXd> derivatives;
        // start with derivative of cost function wrt ML output times dyhat/dprogram output, which
        // is equal to the weight the model assigned to this subprogram (Beta)
        // push back derivative of cost function wrt ML output
        /* cout << "Beta: " << Beta << "\n"; */ 
        derivatives.push_back(this->d_cost_func(y, yhat, sw).array() * Beta*phi.array()); 
        /* cout << "Cost derivative: " << this->d_cost_func(y, f_stack[f_stack.size() - 1]) << "\n"; 
        // Working according to test program */
        /* pop<ArrayXd>(&f_stack); // Get rid of input to cost function */
        vector<BP_NODE> executing; // Stores node and its associated derivatves
        // Currently I don't think updates will be saved, might want a pointer of nodes so don't 
        // have to restock the list
        // Program we loop through and edit during algorithm (is this a shallow or deep copy?)
        /* cout << "copy program \n"; */
        vector<Node*> bp_program = program.get_data(start, end);         
        /* cout << "Initializing backprop systems.\n"; */
        while (bp_program.size() > 0) {
            /* cout << "Size of program: " << bp_program.size() << "\n"; */
            Node* node = pop<Node*>(&bp_program);
            /* cout << "(132) Evaluating: " << node->name << "\n"; */
            /* print_weights(program); */

            vector<ArrayXd> n_derivatives;

            if (isNodeDx(node) && node->visits == 0 && node->arity['f'] > 0) {
                NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
                /* cout << "evaluating derivative\n"; */
                // Calculate all the derivatives and store them, then update all the weights and throw away the node
                for (int i = 0; i < node->arity['f']; i++) {
                    dNode->derivative(n_derivatives, f_stack, i);
                }
                /* cout << "updating derivatives\n"; */
                dNode->update(derivatives, f_stack, this->n);
                // dNode->print_weight();
                /* cout << "popping input arguments\n"; */
                // Get rid of the input arguments for the node
                for (int i = 0; i < dNode->arity['f']; i++) {
                    pop<ArrayXd>(&f_stack);
                }

                if (!n_derivatives.empty()) {
                    derivatives.push_back(pop_front<ArrayXd>(&n_derivatives));
                }

                executing.push_back({dNode, n_derivatives});
            }
            /* cout << "next branch\n"; */
            // Choosing how to move through tree
            if (node->arity['f'] == 0 || !isNodeDx(node)) {
                /* // if this node has arguments on the stack, pop them? */ 
                /* for (int i = 0; i < Node->arity['f']; i++) */ 
                /*     pop<ArrayXd>(&f_stack); */
                
                // Clean up gradients and find the parent node
                pop<ArrayXd>(&derivatives);	// TODO check if this fixed
                next_branch(executing, bp_program, derivatives);
            } else {
                node->visits += 1;
                if (node->visits > node->arity['f']) {
                    next_branch(executing, bp_program, derivatives);
                }
            }
        }

        // point bp_program to null
        for (unsigned i = 0; i < bp_program.size(); ++i)
            bp_program[i] = nullptr;

        /* cout << "Backprop terminated\n"; */
        //print_weights(program);
    }
}

#endif
