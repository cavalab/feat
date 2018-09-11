#include "testsHeader.h"

TEST(NodeTest, Evaluate)
{
	vector<ArrayXd> output;
	ArrayXd x;
	ArrayXb z;
	std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z1;
	
	Stacks stack;
	
	MatrixXd X(3,2); 
    X << -2.0, 0.0, 10.0,
    	 1.0, 0.0, 0.0;
    	 
    ArrayXb Z1(3); 
    ArrayXb Z2(3); 
    Z1 << true, false, true;
    Z2 << true, false, false;
    	 
    X.transposeInPlace();
    
    VectorXd Y(6); 
    Y << 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    
    Data data(X, Y, z1);
    
	std::unique_ptr<Node> addObj = std::unique_ptr<Node>(new NodeAdd());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	addObj->evaluate(data, stack);	
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> subObj = std::unique_ptr<Node>(new NodeSubtract());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	subObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> mulObj = std::unique_ptr<Node>(new NodeMultiply());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	mulObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> divObj = std::unique_ptr<Node>(new NodeDivide());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	divObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> sqrtObj = std::unique_ptr<Node>(new NodeSqrt());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	sqrtObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> sinObj = std::unique_ptr<Node>(new NodeSin());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	sinObj->evaluate(data, stack);	
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> cosObj = std::unique_ptr<Node>(new NodeCos());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	cosObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> squareObj = std::unique_ptr<Node>(new NodeSquare());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	squareObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> expObj = std::unique_ptr<Node>(new NodeExponent());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	expObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> exptObj = std::unique_ptr<Node>(new NodeExponential());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	exptObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> logObj = std::unique_ptr<Node>(new NodeLog());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	logObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> andObj = std::unique_ptr<Node>(new NodeAnd());
	
	stack.f.clear();
	stack.b.clear();
	stack.b.push(Z1);
	stack.b.push(Z2);
	
	andObj->evaluate(data, stack);
	
	z = stack.b.pop();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> orObj = std::unique_ptr<Node>(new NodeOr());
	
	stack.f.clear();
	stack.b.clear();
	stack.b.push(Z1);
	stack.b.push(Z2);
	
	orObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> notObj = std::unique_ptr<Node>(new NodeNot());
	
	stack.f.clear();
	stack.b.clear();
	stack.b.push(Z1);
	
	notObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> eqObj = std::unique_ptr<Node>(new NodeEqual());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	eqObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	 
	std::unique_ptr<Node> gtObj = std::unique_ptr<Node>(new NodeGreaterThan());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	gtObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> geqObj = std::unique_ptr<Node>(new NodeGEQ());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	geqObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> ltObj = std::unique_ptr<Node>(new NodeLessThan());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	ltObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> leqObj = std::unique_ptr<Node>(new NodeLEQ());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	leqObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> ifObj = std::unique_ptr<Node>(new NodeIf());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.b.push(Z1);
	
	ifObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> iteObj = std::unique_ptr<Node>(new NodeIfThenElse());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	stack.b.push(Z1);
	
	iteObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::unique_ptr<Node> tanObj = std::unique_ptr<Node>(new NodeTanh());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	tanObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> logitObj = std::unique_ptr<Node>(new NodeLogit());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	logitObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> stepObj = std::unique_ptr<Node>(new NodeStep());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	stepObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> signObj = std::unique_ptr<Node>(new NodeSign());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	signObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::unique_ptr<Node> xorObj = std::unique_ptr<Node>(new NodeXor());
	
	stack.f.clear();
	stack.b.clear();
	stack.b.push(Z1);
	stack.b.push(Z2);
	
	xorObj->evaluate(data, stack);
	
	z = stack.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> gausObj = std::unique_ptr<Node>(new NodeGaussian());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	
	gausObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> gaus2dObj = std::unique_ptr<Node>(new Node2dGaussian());
	
	stack.f.clear();
	stack.b.clear();
	stack.f.push(X.row(0));
	stack.f.push(X.row(1));
	
	gaus2dObj->evaluate(data, stack);
	
	x = stack.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	//TODO NodeVariable, NodeConstant(both types)
}

