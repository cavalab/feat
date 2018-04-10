#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>

#include "node/nodeDx.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

namespace FT {
	class Auto_backprop {
	public:
		typedef void (*callback)(ArrayXd, ArrayXd);
		// Add a proper constructor
		Auto_backprop(vector<Node> program, callback cost_func, MatrixXd X, VectorXd labels, int iters=1000, double n=0.1) {
			this.program = program;
			this.cost_func = cost_func;
			this.X = X;
			this.labels = labels;
			this.iters = iters;
			this.n = n;
		}

		// Return vector of nodes
		vector<Node> run();

	private:
		double n; // Learning rate
		void*(cost_func)(VectorXd, VectorXd);
		MatrixXd X;
		VectorXd labels;
		int iters;
		vector<Node> program;

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		// Return the f_stack
		void forward_prop();

		// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE> executing, vector<Node> bp_program, vector<ArrayXd> derivatives);

		// Compute gradients and update weights 
		void backprop(vector<ArrayXd> f_stack);

		bool isNodeDx(Node n);

	};

	///////////////////////////////// Definitions
	vector<NodeDx> Auto_backprop::run() {
		// Computes weights via backprop
        for (int x = 0; x < iters; x++) {
        	// Evaluate forward pass
        	vector<ArrayXd> fwd_stack = forward_prop();
            if ((x % round(self.iters/4)) == 0 or x == iters - 1) {
            	print("Currently on iter: " + str(x)); // TODO change to C++ print statement
            }
            
            // Evaluate backward pass
            self.backprop(fwd_stack);
        }            
            
            
        print("Gradient Descent Complete ------------------------------")
        return program

	}

	vector<ArrayXd> Auto_backprop::forward_prop() {
		// Iterate through all the nodes evaluating and tracking ouputs
		vector<ArrayXd> fwd_stack; // Tracks output values
		vector<ArrayXd> execution_stack; // Tracks output values and groups them based on input to functions

		// Use fwd_stack and execution stack to avoid issue of branches affecting what elements appear before a node 
		for (NodeDx p : this.program) { // Can think about changing with call to Node for cases with nodes that aren't differentiable
			for (int i = 0; i < p.arity['f']; i++) {
				fwd_stack.push_back(execution_stack[execution_stack.size() - (p.arity['f'] - i)]);
			}

			p.evaluate(this.X, NULL, execution_stack, NULL);
		}

		fwd_stack.push_back(execution_stack.pop());

		return fwd_stack;
	}

	void next_branch(vector<BP_NODE> executing, vector<Node> bp_program, vector<ArrayXd> derivatives) {
		if(executing.empty()) {
			n_derivatives = []
			BP_NODE bp_node;
            while (bp_node.deriv_list.empty()) {
                bp_node = executing.pop();
                derivatives.pop();
                
                if (executing.empty()) {
                    return NULL;
                }
            }
            
            // Should now have the next parent node
            bp_program.push_back(node.n);
            derivatives.push_back(node.deriv_list.pop(0));  // TODO update this line
            executing.push_back(node);
		}
	}

	void backprop(vector<ArrayXd> f_stack) {
		vector<ArrayXd> derivatives;
		// derivatives.push_back() Might need a cost function node

		vector<BP_NODE> executing; // Stores node and its associated derivatves
		vector<Node> bp_program(this.program); // Program we loop through and edit during algorithm (is this a shallow or deep copy?)

		while (bp_program.size() > 0) {
			Node node = bp_program.pop();

			vector<ArrayXd> n_derivatves;

			if (isNodeDx(node) && node.visits == 0 && node.arity['f'] > 0) {
				NodeDx* dNode = dynamic_cast<NodeDx*>(node);
				// Calculate all the derivatives and store them, then update all the weights and throw away the node
				for (int i = 0; i < node.arity['f']; i++) {
					dNode->derivative(n_derivatives, fwd_stack, node.arity['f'] + i);
				}

				dNode->update(derivatives, fwd_stack, this.n);

				// Get rid of the input arguments for the node
				for (int i = 0; i < dNode->arity['f']; i++) {
					fwd_stack.pop();
				}

				if (n_derivatives.size()) {
					derivatives.push_back(n_derivatives.pop());
				}

				executing.push_back({dNode, n_derivatives});
			}

			// Choosing how to move through tree
			if (node.arity == 0) {
				// Clean up gradients and find the parent node
				next_branch(executing, bp_program, derivatives);
			} else {
				node.visits += 1;
				if (node.visits > node.arity['f']) {
					next_branch(executing, bp_program, derivatives);
				}
			}
		}
	}

	bool isNodeDx(Node n) {
		return d != dynamic_cast<NodeDx*>(n); ;
	}
}

#endif