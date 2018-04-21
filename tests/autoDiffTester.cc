#include <vector>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <math.h>

// Include node and node children
#include "../src/node/node.h"
#include "../src/node/nodeDx.h"
#include "../src/node/nodeadd.h"
#include "../src/node/nodecos.h"
#include "../src/node/nodecube.h"
#include "../src/node/nodedivide.h"
#include "../src/node/nodeexponent.h"
#include "../src/node/nodemultiply.h"
#include "../src/node/nodeexponential.h"
#include "../src/node/nodegaussian.h"
#include "../src/node/nodelog.h"
#include "../src/node/nodelogit.h"
#include "../src/node/noderelu.h"
#include "../src/node/nodesqrt.h"
#include "../src/node/nodesin.h"
#include "../src/node/nodesquare.h"
#include "../src/node/nodesubtract.h"
#include "../src/node/nodetanh.h"
#include "../src/node/nodevariable.h"

// Backprop progam
#include "../src/auto_backprop.h"

// Cost function
#include "../src/metrics.h"

// TODO - implement a testing method that prints out the derivatives for testing against outputs of tensorflow/pythong impl

using namespace std;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using FT::Node;
using FT::NodeDx;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

template <class G>
G pop(vector<G> v) {
	G value = v.back();
	v.pop_back();
	return value;
}

template <class T>
T pop_front(vector<T> v) {
	T value = v.front();
	v.erase(v.begin());
	return value;
}

bool isNodeDx(Node* n) {
	return NULL != dynamic_cast<NodeDx*>(n); 
}

ArrayXd evaluateProgram(vector<Node*> program, MatrixXd data, VectorXd labels) {
	// Create stack for storing execution
	vector<ArrayXd> stack_f;
	vector<ArrayXb> stack_b;

	// Iterate through program and calculate results 
	for (Node* n : program) {
		n->evaluate(data, labels, stack_f, stack_b);
	}

	return pop(stack_f);
}

int testNodes() {
	std::cout << "Starting tests\n";
	vector<ArrayXd> inputs;
	ArrayXd input1(5,1);
	input1(0,0) = 0;
	input1(1,0) = 1;
	input1(2,0) = 2;
	input1(3,0) = 3;
	input1(4,0) = 4;
	ArrayXd input2(5,1);
	input2(0,0) = 4;
	input2(1,0) = 3;
	input2(2,0) = 2;
	input2(3,0) = 1;
	input2(4,0) = 0;
	inputs.push_back(input1);
	inputs.push_back(input2);
	std::cout << "Initialized input vectors.\n";

	// ADD NODE CHECK -------------------------------------------------------------------------------
	NodeDx* toTest = new FT::NodeAdd();
	// Derivative wtr to first input
	ArrayXd expectedDerivative(5, 1);
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	std::cout << "Initialized derivatves.\n";
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Add node failed! (wtr to first input)\n";
	}

	// Derivative wtr to second input
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Add node (wtr to second input) failed!\n";
	}

	// Derivative wtr to first weight
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 3).matrix()).norm() > 0.0001) {
		std::cout << "Add node (wtr to weight on first input) failed!\n";
	}

	// Derivative wtr to second weight
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 2).matrix()).norm() > 0.0001) {
		std::cout << "Add node (wtr to weight on second input) failed!\n";
	}

	// SUB NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSubtract();
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Subtract node (wtr to first input) failed!\n";
	}

	expectedDerivative(0,0) = -1;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -1;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -1;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Subtract node (wtr to second input) failed!\n";
	}

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 3).matrix()).norm() > 0.0001) {
		std::cout << "Subtract node (wtr to weight on first input) failed!\n";
	}

	expectedDerivative(0,0) = -4;
	expectedDerivative(1,0) = -3;
	expectedDerivative(2,0) = -2;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 2).matrix()).norm() > 0.0001) {
		std::cout << "Subtract node (wtr to weight on second input) failed!\n";
	}

	// MULT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeMultiply();
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Multiply node (wtr to first input) failed!\n";
	}

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Multiply node (wtr to second input) failed!\n";
	}

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 4;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 2).matrix()).norm() > 0.0001) {
		std::cout << "Multiply node (wtr to weight on first input) failed!\n";
	}

	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 3).matrix()).norm() > 0.0001) {
		std::cout << "Multiply node (wtr to weight on second input) failed!\n";
	}

	// DIV NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeDivide();
	expectedDerivative(0,0) = MAX_DBL;	// Div by 0 (limited to 0)
	expectedDerivative(1,0) = 1.0/1;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 1.0/4; 
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Divide node (wtr to first input) failed!\n";
	}

	expectedDerivative(0,0) = -MAX_DBL;	// Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/4;
	expectedDerivative(3,0) = -1.0/9;
	expectedDerivative(4,0) = -0.0/16; 
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Divide node (wtr to second input) failed!\n";
	}

	expectedDerivative(0,0) = MAX_DBL;	// Div by 0
	expectedDerivative(1,0) = 3.0/1;
	expectedDerivative(2,0) = 2.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 0.0/4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 3).matrix()).norm() > 0.0001) {
		std::cout << "Divide node (wtr to weight on first input) failed!\n";
	}

	expectedDerivative(0,0) = -MAX_DBL;	//Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/2;
	expectedDerivative(3,0) = -1.0/3;
	expectedDerivative(4,0) = -0.0/4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 2).matrix()).norm() > 0.0001) {
		std::cout << "Divide node (wtr to weight on second input) failed!\n";
	}

	// x^y NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponent();
	expectedDerivative(0,0) = 4 * pow(0,4)/0; // Div by 0
	expectedDerivative(1,0) = 3 * pow(1,3)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/2;
	expectedDerivative(3,0) = 1 * pow(3,1)/3;
	expectedDerivative(4,0) = 0 * pow(4,0)/4;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Exponent node (wtr to first input) failed!\n";
	}

	expectedDerivative(0,0) = 1 * pow(0,4) * log(0); // Log by 0
	expectedDerivative(1,0) = 1 * pow(1,3) * log(1);
	expectedDerivative(2,0) = 1 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(4,0) = 1 * pow(4,0) * log(4);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Exponent node (wtr to second input) failed!\n";
	}

	expectedDerivative(0,0) = 4 * pow(0,4)/1;
	expectedDerivative(1,0) = 3 * pow(1,3)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/1;
	expectedDerivative(3,0) = 1 * pow(3,1)/1;
	expectedDerivative(4,0) = 0 * pow(4,0)/1;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 3).matrix()).norm() > 0.0001) {
		std::cout << "Exponent node (wtr to weight on first input) failed!\n";
	}

	expectedDerivative(0,0) = 4 * pow(0,4) * log(0); // Log by 0
	expectedDerivative(1,0) = 3 * pow(1,3) * log(1);
	expectedDerivative(2,0) = 2 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(4,0) = 0 * pow(4,0) * log(4);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 2).matrix()).norm() > 0.0001) {
		std::cout << "Exponent node (wtr to weight on second input) failed!\n";
	}

	// COS NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeCos();
	expectedDerivative(0,0) = -1 * sin(4);
	expectedDerivative(1,0) = -1 * sin(3);
	expectedDerivative(2,0) = -1 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -1 * sin(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Cos node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = -4 * sin(4);
	expectedDerivative(1,0) = -3 * sin(3);
	expectedDerivative(2,0) = -2 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -0 * sin(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Cos node (wtr to weight) failed!\n";
	}

	// SIN NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSin();
	expectedDerivative(0,0) = 1 * cos(4);
	expectedDerivative(1,0) = 1 * cos(3);
	expectedDerivative(2,0) = 1 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 1 * cos(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Sin node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 4 * cos(4);
	expectedDerivative(1,0) = 3 * cos(3);
	expectedDerivative(2,0) = 2 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 0 * cos(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Sin node (wtr to weight) failed!\n";
	}

	// ^3 NODE CHECK  -------------------------------------------------------------------------------
	toTest = new FT::NodeCube();
	expectedDerivative(0,0) = 3 * pow(4,2);
	expectedDerivative(1,0) = 3 * pow(3,2);
	expectedDerivative(2,0) = 3 * pow(2,2);
	expectedDerivative(3,0) = 3 * pow(1,2);
	expectedDerivative(4,0) = 3 * pow(0,2);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Cube node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 3 * 64;
	expectedDerivative(1,0) = 3 * 27;
	expectedDerivative(2,0) = 3 *  8;
	expectedDerivative(3,0) = 3 *  1;
	expectedDerivative(4,0) = 3 *  0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Cube node (wtr to weight) failed!\n";
	}

	// e^x NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponential();
	expectedDerivative(0,0) = 1 * exp(4);
	expectedDerivative(1,0) = 1 * exp(3);
	expectedDerivative(2,0) = 1 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 1 * exp(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Exponential node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 4 * exp(4);
	expectedDerivative(1,0) = 3 * exp(3);
	expectedDerivative(2,0) = 2 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 0 * exp(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Exponential node (wtr to weight) failed!\n";
	}

	// GAUS NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeGaussian();
	expectedDerivative(0,0) = -2 * 1 * 4 * exp(-16);
	expectedDerivative(1,0) = -2 * 1 * 3 * exp(-9);
	expectedDerivative(2,0) = -2 * 1 * 2 * exp(-4);
	expectedDerivative(3,0) = -2 * 1 * 1 * exp(-1);
	expectedDerivative(4,0) = -2 * 1 * 0 * exp(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) {
		std::cout << "Gaussian node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = -2 * 1 * 16 * exp(-16);
	expectedDerivative(1,0) = -2 * 1 * 9 * exp(-9);
	expectedDerivative(2,0) = -2 * 1 * 4 * exp(-4);
	expectedDerivative(3,0) = -2 * 1 * 1 * exp(-1);
	expectedDerivative(4,0) = -2 * 1 * 0 * exp(0);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) {
		std::cout << "Gaussian node (wtr to weight) failed!\n";
	}

	// LOG NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeLog();
	expectedDerivative(0,0) = 1.0/4;
	expectedDerivative(1,0) = 1.0/3;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = MAX_DBL; // Check if this is intended
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Log node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Log node (wtr to weight) failed!\n";
	}

	// LOGIT NODE CHECK------------------------------------------------------------------------------
	toTest = new FT::NodeLogit();
	expectedDerivative(0,0) = (1 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (1 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (1 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (1 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Logit node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = (4 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (3 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (2 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (0 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Logit node (wtr to weight) failed!\n";
	}

	// RELU NODE CHECK------------------------------------------------------------------------------
	// TODO
	// toTest = new FT::NodeRelu();
	// expectedDerivative(0,0) = 1;
	// expectedDerivative(1,0) = 1;
	// expectedDerivative(2,0) = 1;
	// expectedDerivative(3,0) = 1;
	// expectedDerivative(4,0) = 1;
	// if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
	// 	std::cout << "Relu node failed!\n";
	// }

	// SQRT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeSqrt();
	expectedDerivative(0,0) = 1/(2 * sqrt(4));
	expectedDerivative(1,0) = 1/(2 * sqrt(3));
	expectedDerivative(2,0) = 1/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 1/(2 * sqrt(0));
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Sqrt node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 4/(2 * sqrt(4));
	expectedDerivative(1,0) = 3/(2 * sqrt(3));
	expectedDerivative(2,0) = 2/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 0/(2 * sqrt(0));
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Sqrt node(wtr to weight) failed!\n";
	}

	// ^2  NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSquare();
	expectedDerivative(0,0) = 2 * 1 * 4;
	expectedDerivative(1,0) = 2 * 1 * 3;
	expectedDerivative(2,0) = 2 * 1 * 2;
	expectedDerivative(3,0) = 2 * 1 * 1;
	expectedDerivative(4,0) = 2 * 1 * 0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Square node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 2 * 16;
	expectedDerivative(1,0) = 2 *  9;
	expectedDerivative(2,0) = 2 *  4;
	expectedDerivative(3,0) = 2 *  1;
	expectedDerivative(4,0) = 2 *  0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Square node (wtr to weight) failed!\n";
	}

	// TANH NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeTanh();
	expectedDerivative(0,0) = 0.0013409506830258968799702;
	expectedDerivative(1,0) = 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 1;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Tanh node (wtr to input) failed!\n";
	}

	expectedDerivative(0,0) = 4 * 0.0013409506830258968799702;
	expectedDerivative(1,0) = 3 * 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 2 * 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 1 * 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 0;
	if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 1).matrix()).norm() > 0.0001) { // Currently pseudocode
		std::cout << "Tanh node (wtr to weight) failed!\n";
	}
}

int testDummyProgram() {
	// Create a vector of nodes as a dummy program
	vector<Node*> p0;
	p0.push_back(new FT::NodeVariable(1));
	p0.push_back(new FT::NodeVariable(2));
	p0.push_back(new FT::NodeAdd());
	p0.push_back(new FT::NodeVariable(1));
	p0.push_back(new FT::NodeTanh());
	p0.push_back(new FT::NodeSin());

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


	// Auto_backprop(PROGRAM, COST_FUNCTION, D_COST_FUNCTION, INPUT_DATA, LABELS, ITERS, LEARNING RATE);
	FT::Auto_backprop* engine = new FT::Auto_backprop(p0, NULL, x, y, iters, learning_rate);
	vector<Node*> predictor = engine->run();

	// Test if output is correct
	ArrayXd pred = evaluateProgram(predictor, x, y);
	ArrayXd target;
	target << 0; // Populate with expected results as dictated by tensorflow

	// Test if weights are correct
	vector<double> expected_weights = {};

	int count = 0;
	for (int i = 0; i < predictor.size(); i++) { // Need check if node doesn't have weights
		// Check if node is differentiable
		if (isNodeDx(predictor[i])) {
			if (abs(expected_weights[count] - dynamic_cast<NodeDx*>(predictor[i])->W[0]) > 0.00001) {
				cout << "Discrepency with node " << i << "\n";
			}
			count++;
		}
	}
}

int main() {
	testNodes();
	return 1;
}