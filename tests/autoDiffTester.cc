#include <vector>
#include <iostream>
#include <string>
#include <stack>
#include <Eigen/Dense>

// Include node and node children
#include "../src/node/node.h"
#include "../src/node/nodeDx.h"
#include "../src/node/nodeadd.h"
#include "../src/node/nodecos.h"
#include "../src/node/nodecube.h"
#include "../src/node/nodedivide.h"
#include "../src/node/nodeexponent.h"
#include "../src/node/nodeexponential.h"
#include "../src/node/nodegaussian.h"
#include "../src/node/nodelog.h"
#include "../src/node/nodelogit.h"
#include "../src/node/nodemultiply.h"
#include "../src/node/noderelu.h"
#include "../src/node/noderoot.h"
#include "../src/node/nodesqrt.h"
#include "../src/node/nodesquare.h"
#include "../src/node/nodesubtract.h"
#include "../src/node/nodetanh.h"

// Backprop progam
#include "../src/auto_backprop.h"


using namespace std;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

int main() {
	// testNodes();
	return 1;
}

int testNodes() {
	NodeDx toTest = new NodeAdd();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Add node failed!\n";
	}

	toTest = new NodeSubtract();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Subtract node failed!\n";
	}

	toTest = new nodemultiply();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Multiply node failed!\n";
	}

	toTest = new NodeDivide();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Divide node failed!\n";
	}

	toTest = new NodeExponent();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Exponent node failed!\n";
	}

	toTest = new NodeCos();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Cos node failed!\n";
	}

	toTest = new NodeSin();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Sin node failed!\n";
	}

	toTest = new NodeCube();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Cube node failed!\n";
	}

	toTest = new NodeExponential();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Exponential node failed!\n";
	}

	toTest = new NodeGaussian();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Gaussian node failed!\n";
	}

	toTest = new NodeLog();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Log node failed!\n";
	}

	toTest = new NodeLogit();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Logit node failed!\n";
	}

	toTest = new NodeRelu();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Relu node failed!\n";
	}

	toTest = new NodeSqrt();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Sqrt node failed!\n";
	}

	toTest = new NodeSquare();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Square node failed!\n";
	}

	toTest = new NodeTanh();
	expectedDerivative = NULL;
	if (toTest.getDerivative != expectedDerivative) { // Currently pseudocode
		std::cout << "Tanh node failed!\n";
	}
}

int testDummyProgram() {
	// Create a vector of nodes as a dummy program
	vector<Node> p0;
	p0.push_back(new NodeVariable(1));
	p0.push_back(new NodeVariable(2));
	p0.push_back(new NodeAdd());
	p0.push_back(new NodeVariable(1));
	p0.push_back(new NodeTanh());
	p0.push_back(new NodeSin());

	// Create cost function 

	// Create input data and labels
	MatrixXd x(10, 2);
	VectorXd y(10);
	x << 0.0, -9.0,
	     1.0, -8.0,
	     2.0, -7.0,
	     3.0, -6.0,
	     4.0, -5.0,
	     5.0, -4.0,
	     6.0, -3.0,
	     7.0, -2.0,
	     8.0, -1.0,
	     9.0, -0.0;

	y << 9.0,
		 8.0,
		 7.0,
		 6.0,
		 5.0,
		 4.5,
		 3.0,
		 2.0,
		 1.0,
		 0.0;

	// Params
	int iters = 500;
	double learning_rate = 0.01;


	// Auto_backprop(PROGRAM, COST_FUNCTION, INPUT_DATA, LABELS, ITERS, LEARNING RATE);
	Auto_backprop engine = new Auto_backprop(p0, COST_FUNCTION, x, y, iters, learning_rate);
	vector<Node> predictor = engine.run();
}

int testSimpleBackProp() {

}