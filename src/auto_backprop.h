#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>

#include "node/node.h"
#include "node/nodeDx.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

/**
TODOS
------------------------
Integrate cost function
Integrate vectorList
Integrate pointers?
More graceful handling of non derivative nodes (like adding one to derivative stack and such)
**/

namespace FT {
	class Auto_backprop {
	public:
		typedef void (*callback)(ArrayXd, ArrayXd);
		// Add a proper constructor
		// Auto_backprop(vector<Node> program, callback cost_func, MatrixXd X, VectorXd labels, int iters=1000, double n=0.1) {
		// 	this->program = program;
		// 	this->cost_func = cost_func;
		// 	this->X = X;
		// 	this->labels = labels;
		// 	this->iters = iters;
		// 	this->n = n;
		// }

		// Return vector of nodes
		// vector<Node> run();

	private:
		double n; // Learning rate
		// void*(cost_func)(VectorXd, VectorXd);
		callback cost_func;
		MatrixXd X;
		VectorXd labels;
		int iters;
		vector<Node*> program;

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		// Return the f_stack
		vector<ArrayXd> forward_prop() 
		{
			// Iterate through all the nodes evaluating and tracking ouputs
			vector<ArrayXd> stack_f; // Tracks output values
			vector<ArrayXd> execution_stack; // Tracks output values and groups them based on input to functions
			vector<ArrayXb> tmp;

			// Use stack_f and execution stack to avoid issue of branches affecting what elements appear before a node 
			for (Node* p : this->program) { // Can think about changing with call to Node for cases with nodes that aren't differentiable
				for (int i = 0; i < p->arity['f']; i++) {
					stack_f.push_back(execution_stack[execution_stack.size() - (p->arity['f'] - i)]);
				}

				p->evaluate(this->X, this->labels, execution_stack, tmp);
			}

			stack_f.push_back(pop<ArrayXd>(execution_stack)); // Would be nice to create a general "pop" function that does both these steps at once

			return stack_f;
		}

		// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE> executing, vector<Node*> bp_program, vector<ArrayXd> derivatives) {
			// While there are still nodes with branches to explore
			if(!executing.empty()) {
				// Declare variable to hold node and its associated derivatives
				BP_NODE bp_node = pop<BP_NODE>(executing); // Check first element
	            // Loop until branch to explore is found
	            while (bp_node.deriv_list.empty()) {
	                bp_node = pop<BP_NODE>(executing); // Get node and its derivatves
	                derivatives.pop_back(); // Remove associated gradients from stack
	                
	                if (executing.empty()) {
	                    return;
	                }
	            }
	            
	            // Should now have the next parent node and derivatves (stored in bp_node)
	            bp_program.push_back(bp_node.n);
	            derivatives.push_back(pop_front<ArrayXd>(bp_node.deriv_list)); // Pull derivative from front of list due to how we stored them earlier
	            executing.push_back(bp_node); // Push it back on the stack in order to sync all the stacks
			}
		}

		// Compute gradients and update weights 
		void backprop(vector<ArrayXd> f_stack) {
			vector<ArrayXd> derivatives;
			// derivatives.push_back() Might need a cost function node (still need to address this)

			vector<BP_NODE> executing; // Stores node and its associated derivatves
			// Currently I don't think updates will be saved, might want a pointer of nodes so don't have to restock the list
			vector<Node*> bp_program(this->program); // Program we loop through and edit during algorithm (is this a shallow or deep copy?)

			while (bp_program.size() > 0) {
				Node* node = pop<Node*>(bp_program);

				vector<ArrayXd> n_derivatives;

				if (isNodeDx(node) && node->visits == 0 && node->arity['f'] > 0) {
					NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
					// Calculate all the derivatives and store them, then update all the weights and throw away the node
					for (int i = 0; i < node->arity['f']; i++) {
						dNode->derivative(n_derivatives, f_stack, node->arity['f'] + i);
					}

					dNode->update(derivatives, f_stack, this->n);

					// Get rid of the input arguments for the node
					for (int i = 0; i < dNode->arity['f']; i++) {
						pop<ArrayXd>(f_stack);
					}

					if (!n_derivatives.empty()) {
						derivatives.push_back(pop_front<ArrayXd>(n_derivatives));
					}

					executing.push_back({dNode, n_derivatives});
				}

				// Choosing how to move through tree
				if (node->arity['f'] == 0) {
					// Clean up gradients and find the parent node
					next_branch(executing, bp_program, derivatives);
				} else {
					node->visits += 1;
					if (node->visits > node->arity['f']) {
						next_branch(executing, bp_program, derivatives);
					}
				}
			}
		}

		bool isNodeDx(Node* n) {
			return NULL != dynamic_cast<NodeDx*>(n); ;
		}

		template <class T>
		T pop(vector<T> v) {
			T value = v.back();
			v.pop_back();
			return value;
		}

		template <class T>
		T pop_front(vector<T> v) {
			T value = v.front();
			v.erase(0);
			return value;
		}

	public:
		Auto_backprop(vector<Node*> program, callback cost_func, MatrixXd X, VectorXd labels, int iters=1000, double n=0.1) {
			this->program = program;
			this->cost_func = cost_func;
			this->X = X;
			this->labels = labels;
			this->iters = iters;
			this->n = n;
		}

		vector<Node*> run() 
		{
			// Computes weights via backprop
	        for (int x = 0; x < this->iters; x++) {
	        	// Evaluate forward pass
	        	vector<ArrayXd> stack_f = forward_prop();
	            // if ((x % round(this->iters/4)) == 0 or x == this->iters - 1) {
	            // 	std::cout << "Iters are happening!\n"; // print("Currently on iter: " + str(x)); // TODO change to C++ print statement
	            // }
	            
	            // Evaluate backward pass
	            backprop(stack_f);
	        }
	        return program;
		}
	};

	///////////////////////////////// Definitions
	// vector<Node> Auto_backprop::run() {
	// 	// Computes weights via backprop
 //        for (int x = 0; x < this->iters; x++) {
 //        	// Evaluate forward pass
 //        	vector<ArrayXd> stack_f = forward_prop();
 //            if ((x % round(this->iters/4)) == 0 or x == this->iters - 1) {
 //            	std::cout << "Iters are happening!\n"; // print("Currently on iter: " + str(x)); // TODO change to C++ print statement
 //            }
            
 //            // Evaluate backward pass
 //            self.backprop(stack_f);
 //        }
 //        return program;
	// }

	// vector<ArrayXd> Auto_backprop::forward_prop() {
	// 	// Iterate through all the nodes evaluating and tracking ouputs
	// 	vector<ArrayXd> stack_f; // Tracks output values
	// 	vector<ArrayXd> execution_stack; // Tracks output values and groups them based on input to functions

	// 	// Use stack_f and execution stack to avoid issue of branches affecting what elements appear before a node 
	// 	for (Node p : this->program) { // Can think about changing with call to Node for cases with nodes that aren't differentiable
	// 		for (int i = 0; i < p.arity['f']; i++) {
	// 			stack_f.push_back(execution_stack[execution_stack.size() - (p.arity['f'] - i)]);
	// 		}

	// 		p.evaluate(this->X, NULL, execution_stack, NULL);
	// 	}

	// 	stack_f.push_back(pop<ArrayXd>(execution_stack)); // Would be nice to create a general "pop" function that does both these steps at once

	// 	return stack_f;
	// }

	// void Auto_backprop::next_branch(vector<BP_NODE> executing, vector<Node> bp_program, vector<ArrayXd> derivatives) {
	// 	// While there are still nodes with branches to explore
	// 	if(!executing.empty()) {
	// 		// Declare variable to hold node and its associated derivatives
	// 		BP_NODE bp_node = pop<BP_NODE>(executing); // Check first element
 //            // Loop until branch to explore is found
 //            while (bp_node.deriv_list.empty()) {
 //                bp_node = pop<BP_NODE>(executing); // Get node and its derivatves
 //                derivatives.pop_back(); // Remove associated gradients from stack
                
 //                if (executing.empty()) {
 //                    return NULL;
 //                }
 //            }
            
 //            // Should now have the next parent node and derivatves (stored in bp_node)
 //            bp_program.push_back(bp_node.n);
 //            derivatives.push_back(pop_front<ArrayXd>(node.deriv_list); // Pull derivative from front of list due to how we stored them earlier
 //            executing.push_back(bp_node); // Push it back on the stack in order to sync all the stacks
	// 	}
	// }

	// void Auto_backprop::backprop(vector<ArrayXd> f_stack) {
	// 	vector<ArrayXd> derivatives;
	// 	// derivatives.push_back() Might need a cost function node (still need to address this)

	// 	vector<BP_NODE> executing; // Stores node and its associated derivatves
	// 	// Currently I don't think updates will be saved, might want a pointer of nodes so don't have to restock the list
	// 	vector<Node> bp_program(this->program); // Program we loop through and edit during algorithm (is this a shallow or deep copy?)

	// 	while (bp_program.size() > 0) {
	// 		Node node = pop<Node>(bp_program);

	// 		vector<ArrayXd> n_derivatives;

	// 		if (isNodeDx(&node) && node.visits == 0 && node.arity['f'] > 0) {
	// 			NodeDx* dNode = dynamic_cast<NodeDx*>(&node); // Could probably put this up one and have the if condition check if null
	// 			// Calculate all the derivatives and store them, then update all the weights and throw away the node
	// 			for (int i = 0; i < node.arity['f']; i++) {
	// 				dNode->derivative(n_derivatives, f_stack, node.arity['f'] + i);
	// 			}

	// 			dNode->update(derivatives, f_stack, this->n);

	// 			// Get rid of the input arguments for the node
	// 			for (int i = 0; i < dNode->arity['f']; i++) {
	// 				pop<ArrayXd>(f_stack);
	// 			}

	// 			if (!n_derivatives.empty() {
	// 				derivatives.push_back(pop_front<ArrayXd>(n_derivatives));
	// 			}

	// 			executing.push_back({dNode, n_derivatives});
	// 		}

	// 		// Choosing how to move through tree
	// 		if (node.arity['f'] == 0) {
	// 			// Clean up gradients and find the parent node
	// 			next_branch(executing, bp_program, derivatives);
	// 		} else {
	// 			node.visits += 1;
	// 			if (node.visits > node.arity['f']) {
	// 				next_branch(executing, bp_program, derivatives);
	// 			}
	// 		}
	// 	}
	// }

	// bool Auto_backprop::isNodeDx(Node* n) {
	// 	return NULL != dynamic_cast<NodeDx*>(n); ;
	// }

	// T Auto_backprop::pop(vector<T> v) {
	// 	T value = v.back();
	// 	v.pop_back();
	// 	return value;
	// }

	// T Auto_backprop::pop_front(vector<T> v) {
	// 	T value = v.front();
	// 	v.erase(0);
	// 	return value;
	// }
}

#endif