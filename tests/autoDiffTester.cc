#include <vector>
#include <iostream>
#include <string>
#include <stack>
#include <../src/node/node.h>
#include <../src/node/nodeDx.h>
#include <../src/node/nodeadd.h>
#include <../src/node/nodecos.h>
#include <../src/node/nodecube.h>
#include <../src/node/nodedivide.h>
#include <../src/node/nodeexponent.h>
#include <../src/node/nodeexponential.h>
#include <../src/node/nodegaussian.h>
#include <../src/node/nodelog.h>
#include <../src/node/nodelogit.h>
#include <../src/node/nodemultiply.h>
#include <../src/node/noderelu.h>
#include <../src/node/noderoot.h>
#include <../src/node/nodesqrt.h>
#include <../src/node/nodesquare.h>
#include <../src/node/nodesubtract.h>
#include <../src/node/nodetanh.h>


using namespace std;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

int main() {
	testNodes();
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
}

int testDummyProgram() {

}

int testSimpleBackProp() {

}