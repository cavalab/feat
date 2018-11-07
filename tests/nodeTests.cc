#include "testsHeader.h"

TEST(NodeTest, Evaluate)
{
	vector<ArrayXd> output;
	ArrayXd x;
	ArrayXb z;
	std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z1;
	
	State state;
	
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
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	addObj->evaluate(data, state);	
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> subObj = std::unique_ptr<Node>(new NodeSubtract());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	subObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> mulObj = std::unique_ptr<Node>(new NodeMultiply());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	mulObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> divObj = std::unique_ptr<Node>(new NodeDivide());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	divObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> sqrtObj = std::unique_ptr<Node>(new NodeSqrt());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	sqrtObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> sinObj = std::unique_ptr<Node>(new NodeSin());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	sinObj->evaluate(data, state);	
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> cosObj = std::unique_ptr<Node>(new NodeCos());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	cosObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> squareObj = std::unique_ptr<Node>(new NodeSquare());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	squareObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> expObj = std::unique_ptr<Node>(new NodeExponent());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	expObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> exptObj = std::unique_ptr<Node>(new NodeExponential());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	exptObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> logObj = std::unique_ptr<Node>(new NodeLog());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	logObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> andObj = std::unique_ptr<Node>(new NodeAnd());
	
	state.f.clear();
	state.b.clear();
	state.b.push(Z1);
	state.b.push(Z2);
	
	andObj->evaluate(data, state);
	
	z = state.b.pop();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> orObj = std::unique_ptr<Node>(new NodeOr());
	
	state.f.clear();
	state.b.clear();
	state.b.push(Z1);
	state.b.push(Z2);
	
	orObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> notObj = std::unique_ptr<Node>(new NodeNot());
	
	state.f.clear();
	state.b.clear();
	state.b.push(Z1);
	
	notObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> eqObj = std::unique_ptr<Node>(new NodeEqual());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	eqObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	 
	std::unique_ptr<Node> gtObj = std::unique_ptr<Node>(new NodeGreaterThan());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	gtObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> geqObj = std::unique_ptr<Node>(new NodeGEQ());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	geqObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> ltObj = std::unique_ptr<Node>(new NodeLessThan());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	ltObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> leqObj = std::unique_ptr<Node>(new NodeLEQ());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	leqObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> ifObj = std::unique_ptr<Node>(new NodeIf());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.b.push(Z1);
	
	ifObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> iteObj = std::unique_ptr<Node>(new NodeIfThenElse());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	state.b.push(Z1);
	
	iteObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::unique_ptr<Node> tanObj = std::unique_ptr<Node>(new NodeTanh());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	tanObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> logitObj = std::unique_ptr<Node>(new NodeLogit());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	logitObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> stepObj = std::unique_ptr<Node>(new NodeStep());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	stepObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> signObj = std::unique_ptr<Node>(new NodeSign());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	signObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::unique_ptr<Node> xorObj = std::unique_ptr<Node>(new NodeXor());
	
	state.f.clear();
	state.b.clear();
	state.b.push(Z1);
	state.b.push(Z2);
	
	xorObj->evaluate(data, state);
	
	z = state.b.pop(); 
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::unique_ptr<Node> gausObj = std::unique_ptr<Node>(new NodeGaussian());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	
	gausObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::unique_ptr<Node> gaus2dObj = std::unique_ptr<Node>(new Node2dGaussian());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	gaus2dObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	//TODO NodeVariable, NodeConstant(both types)
}

