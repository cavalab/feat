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
	
        typedef VectorXd (*callback)(const VectorXd&, const VectorXd&);
        
        AutoBackProp(callback d_cost_func, int iters=1000, double n=0.1) 
        {
			/* this->program = program.get_data(); */
			this->d_cost_func = d_cost_func;
			/* this->X = X; */
			/* this->labels = labels; */
			this->iters = iters;
			this->n = n;
		}
        /// adapt weights
		void run(NodeVector& program, MatrixXd& X, VectorXd& y,
                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

        /* ~AutoBackProp() */
        /* { */
        /*     /1* for (const auto& p: program) *1/ */
        /*         /1* p = nullptr; *1/ */
        /* } */


	private:
		double n;                   //< Learning rate
		callback d_cost_func;       //< cost function pointer
		/* MatrixXd X;                 //< Input data */
		/* VectorXd labels;            //< labels */
		int iters;                  //< iterations
		/* vector<Node*> program;      //< program to adapt */

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		void print_weights(NodeVector& program) {
			for (const auto& p : program) {
				cout << "(46) Node: " << p->name;
				if (isNodeDx(p)) {
                    
					NodeDx* dNode = dynamic_cast<NodeDx*>(p.get());
                    cout << " with weight";
					for (int i = 0; i < dNode->arity['f']; i++) {
						cout << " " << dNode->W.at(i);
					}
                    dNode = nullptr;
				}
				cout << "\n";
			}
		}
		/// Return the f_stack
		vector<ArrayXd> forward_prop(NodeVector& program, MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

		/// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                         vector<ArrayXd>& derivatives);

        /// Compute gradients and update weights 
		void backprop(vector<ArrayXd>& f_stack, NodeVector& program, MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

        /// check if differentiable node    
        bool isNodeDx(Node* n){ return NULL != dynamic_cast<NodeDx*>(n); ; }

		bool isNodeDx(const std::unique_ptr<Node>& n) {
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
    void AutoBackProp::run(NodeVector& program, MatrixXd& X, VectorXd& y, 
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z)
    {
        cout << "Starting up AutoBackProp with " << this->iters << " iterations.";
        // Computes weights via backprop
        for (int x = 0; x < this->iters; x++) {
            // Evaluate forward pass
            vector<ArrayXd> stack_f = forward_prop(program, X, y, Z);
            // if ((x % round(this->iters/4)) == 0 or x == this->iters - 1) {
            cout << "Iters are happening!\n"; // print("Currently on iter: " + str(x)); // TODO change to C++ print statement
            // }
            
            // Evaluate backward pass
            backprop(stack_f, program, X, y, Z);
        }
        cout << "Finished backprop. returning program:\n";
        print_weights(program);    
        /* return this->program; */
    }
    
    // Return the f_stack
    vector<ArrayXd> AutoBackProp::forward_prop(NodeVector& program, MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z) 
    {
        cout << "Forward pass\n";
        // Iterate through all the nodes evaluating and tracking ouputs
        vector<ArrayXd> stack_f; // Tracks output values
        vector<ArrayXd> execution_stack; // Tracks output values and groups them based on input to functions
        vector<ArrayXb> tmp;

        FT::Stacks stack;

        // Use stack_f and execution stack to avoid issue of branches affecting what elements appear before a node 
        for (const auto& p : program) 
        { // Can think about changing with call to Node for cases with nodes that aren't differentiable
            cout << "Evaluating node: " << p->name << "\n";
            for (int i = 0; i < p->arity['f']; i++) {
                stack_f.push_back(stack.f.at(stack.f.size() - (p->arity['f'] - i)));
                // stack_f.push_back(execution_stack[execution_stack.size() - (p->arity['f'] - i)]);
            }

            p->evaluate(X, y, Z, stack); // execution_stack, tmp);
            p->visits = 0;
        }

        stack_f.push_back(stack.f.pop());
        //stack_f.push_back(pop<ArrayXd>(&execution_stack)); // Would be nice to create a general "pop" function that does both these steps at once

        cout << "Returning forward pass.\n";
        return stack_f;
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
            if (!bp_node.deriv_list.empty()) {
                bp_program.push_back(bp_node.n);
                derivatives.push_back(pop_front<ArrayXd>(&(bp_node.deriv_list))); // Pull derivative from front of list due to how we stored them earlier
                executing.push_back(bp_node); // Push it back on the stack in order to sync all the stacks
            }
        }
    }

    // Compute gradients and update weights 
    void AutoBackProp::backprop(vector<ArrayXd>& f_stack, NodeVector& program, MatrixXd& X, 
            VectorXd& y, std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z)    
    {
        cout << "---------------------------\n";
        vector<ArrayXd> derivatives;
        derivatives.push_back(this->d_cost_func(y, f_stack[f_stack.size() - 1])); // Might need a cost function node (still need to address this)
        cout << "Cost derivative: " << this->d_cost_func(y, f_stack[f_stack.size() - 1]) << "\n"; // Working according to test program
        cout << "MSE: " << (y.array()-f_stack[f_stack.size() - 1]).array().pow(2).mean() << "\n";
        pop<ArrayXd>(&f_stack); // Get rid of input to cost function

        vector<BP_NODE> executing; // Stores node and its associated derivatves
        // Currently I don't think updates will be saved, might want a pointer of nodes so don't have to restock the list
        vector<Node*> bp_program = program.get_data(); // Program we loop through and edit during algorithm (is this a shallow or deep copy?)

        cout << "Initializing backprop systems.\n";
        while (bp_program.size() > 0) {
            Node* node = pop<Node*>(&bp_program);
            /* cout << "Size of program: " << bp_program.size() << "\n"; */
            /* cout << "(132) Evaluating: " << node->name << "\n"; */
            /* print_weights(program); */

            vector<ArrayXd> n_derivatives;

            if (isNodeDx(node) && node->visits == 0 && node->arity['f'] > 0) {
                NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
                // Calculate all the derivatives and store them, then update all the weights and throw away the node
                for (int i = 0; i < node->arity['f']; i++) {
                    dNode->derivative(n_derivatives, f_stack, i);
                }

                dNode->update(derivatives, f_stack, this->n);
                // dNode->print_weight();

                // Get rid of the input arguments for the node
                for (int i = 0; i < dNode->arity['f']; i++) {
                    pop<ArrayXd>(&f_stack);
                }

                if (!n_derivatives.empty()) {
                    derivatives.push_back(pop_front<ArrayXd>(&n_derivatives));
                }

                executing.push_back({dNode, n_derivatives});
            }

            // Choosing how to move through tree
            if (node->arity['f'] == 0 || !isNodeDx(node)) {
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

        cout << "Backprop terminated\n";
        print_weights(program);
    }
}

#endif
