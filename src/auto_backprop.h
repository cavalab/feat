#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "node/node.h"
#include "node/nodeDx.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

/**
TODO
------------------------
Integrate cost function
Integrate vectorList
Integrate pointers?
More graceful handling of non derivative nodes (like adding one to derivative stack and such)
TODO Integrate derivative of cost function pointer 
TODO Make it so stops traversing once it hits a non-differentiable node and then goes upstream and finds another branch to traverse
**/

namespace FT {
	class Auto_backprop {
	// public:
	typedef VectorXd (*callback)(const VectorXd&, const VectorXd&);

	private:
		double n; // Learning rate
		// void*(cost_func)(VectorXd, VectorXd);
		// callback cost_func;
		// callback d_cost_func;
		callback cost_func; // Not actually necessary 
		callback d_cost_func;
		MatrixXd X;
		VectorXd labels;
		int iters;
		vector<Node*> program;

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		void print_weights() {
			for (Node* p : this->program) {
				std::cout << "Node: " << p->name;
				if (isNodeDx(p)) {
					NodeDx* dNode = dynamic_cast<NodeDx*>(p);
					std::cout << " with weight";
					for (int i = 0; i < dNode->arity['f']; i++) {
						std::cout << " " << dNode->W[i];
					}
				}
				std::cout << "\n";
			}
		}

		// Return the f_stack
		vector<ArrayXd> forward_prop() 
		{
			std::cout << "Forward pass execution running.\n";
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
				p->visits = 0;
			}

			stack_f.push_back(pop<ArrayXd>(&execution_stack)); // Would be nice to create a general "pop" function that does both these steps at once

			std::cout << "Forward pass execution terminated.\n";
			return stack_f;
		}

		// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, vector<ArrayXd>& derivatives) {
			// While there are still nodes with branches to explore
			if(!executing.empty()) {
				// Declare variable to hold node and its associated derivatives
				BP_NODE bp_node = pop<BP_NODE>(&executing); // Check first element
	            // Loop until branch to explore is found
	            while (bp_node.deriv_list.empty() && !executing.empty()) {
	            	std::cout << "Looping\n";
	                bp_node = pop<BP_NODE>(&executing); // Get node and its derivatves
	                std::cout << "recovered node\n";

	                // For some reason this function is not removing element from the stack
	                std::cout << "Before pop: " << derivatives.size() << "\n";
	                pop<ArrayXd>(&derivatives); // Remove associated gradients from stack
	                std::cout << "After pop: " << derivatives.size() << "\n";
	                std::cout << "Checking if\n";
	                if (executing.empty()) {
	                	std::cout << "Returning\n";
	                    return;
	                }
	            }
	            
	            std::cout << "Have parent node\n";
	            std::cout << "Node: " << bp_node.n << "\n";
	            // Should now have the next parent node and derivatves (stored in bp_node)
	            if (!bp_node.deriv_list.empty()) {
	            	bp_program.push_back(bp_node.n);
	            	derivatives.push_back(pop_front<ArrayXd>(&(bp_node.deriv_list))); // Pull derivative from front of list due to how we stored them earlier
	            	executing.push_back(bp_node); // Push it back on the stack in order to sync all the stacks
	            }
			}
			std::cout << "Got to next branch\n";
		}

		// Compute gradients and update weights 
		void backprop(vector<ArrayXd> f_stack) {
			std::cout << "---------------------------\n";
			vector<ArrayXd> derivatives;
			derivatives.push_back(this->d_cost_func(this->labels, f_stack[f_stack.size() - 1])); // Might need a cost function node (still need to address this)
			std::cout << "Cost derivative: " << this->d_cost_func(this->labels, f_stack[f_stack.size() - 1]) << "\n"; // Working according to test program
			pop<ArrayXd>(&f_stack); // Get rid of input to cost function

			vector<BP_NODE> executing; // Stores node and its associated derivatves
			// Currently I don't think updates will be saved, might want a pointer of nodes so don't have to restock the list
			vector<Node*> bp_program(this->program); // Program we loop through and edit during algorithm (is this a shallow or deep copy?)

			// std::cout << "Initializing backprop systems.\n";
			while (bp_program.size() > 0) {
				Node* node = pop<Node*>(&bp_program);
				// std::cout << "Size of program: " << bp_program.size() << "\n";
				std::cout << "Evaluating: " << node->name << "\n";
				// print_weights();

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

				std::cout << "Checking branch\n";
				// Choosing how to move through tree
				if (node->arity['f'] == 0 || !isNodeDx(node)) {
					// Clean up gradients and find the parent node
					pop<ArrayXd>(&derivatives);	// TODO check if this fixed
					next_branch(executing, bp_program, derivatives);
				} else {
					node->visits += 1;
					if (node->visits > node->arity['f']) {
						std::cout << "Going to next branch";
						next_branch(executing, bp_program, derivatives);
					}
				}
			}

			// std::cout << "Backprop terminated\n";
			print_weights();
		}

		bool isNodeDx(Node* n) {
			return NULL != dynamic_cast<NodeDx*>(n); ;
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

	public:
		Auto_backprop(vector<Node*> program, callback cost_func, callback d_cost_func, MatrixXd X, VectorXd labels, int iters=1000, double n=0.1) {
			this->program = program;
			this->cost_func = cost_func;
			this->d_cost_func = d_cost_func;
			this->X = X;
			this->labels = labels;
			this->iters = iters;
			this->n = n;
		}

		vector<Node*> run() 
		{
			std::cout << "Starting up Auto_backprop with " << this->iters << " iterations.";
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
	        return this->program;
		}
	};
}

#endif