#include "testsHeader.h"
#include "cudaTestUtils.h"

#ifndef USE_CUDA
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
	
	/*std::unique_ptr<Node> gaus2dObj = std::unique_ptr<Node>(new Node2dGaussian());
	
	state.f.clear();
	state.b.clear();
	state.f.push(X.row(0));
	state.f.push(X.row(1));
	
	gaus2dObj->evaluate(data, state);
	
	x = state.f.pop(); 
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));*/
	
	//TODO NodeVariable, NodeConstant(both types)
}
#else

void evaluateCudaNodes(NodeVector &nodes, MatrixXd &X, string testNode)
{
    Stacks stack;
    
    VectorXd Y(6); 
    Y << 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z1;
    
    Data data(X, Y, z1);
    
    std::map<char, size_t> stack_size = get_max_stack_size(nodes);
    
    choose_gpu();
    
    stack.allocate(stack_size,data.X.cols());
    
    for (const auto& n : nodes)   
    {
        n->evaluate(data, stack);
        stack.update_idx(n->otype, n->arity);
    }	
    
    stack.copy_to_host();
    
    std::cout<<"Printing output now for test node " << testNode;
    std::cout<<"\n********************************\n";
    
    std::cout<<"Floating stack is\n";
    std::cout<< stack.f << "\n\n";
    
    std::cout<<"Boolean stack is\n";
    std::cout<< stack.b << "\n";
    
    std::cout<<"\n********************************\n";
    
    
}

TEST(NodeTest, Evaluate)
{
    initialize_cuda(); 

	MatrixXd X1(2,3); 
    X1 << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0;   
   
    //vector<std::unique_ptr<Node> > nodes;
    NodeVector nodes;
    
    std::unique_ptr<Node> f1 = std::unique_ptr<Node>(new NodeVariable<double>(0));
    std::unique_ptr<Node> f2 = std::unique_ptr<Node>(new NodeVariable<double>(1));
    
    std::unique_ptr<Node> addObj = std::unique_ptr<Node>(new NodeAdd());

    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(addObj->clone());

    evaluateCudaNodes(nodes, X1, "add");
    
    std::unique_ptr<Node> subObj = std::unique_ptr<Node>(new NodeSubtract());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(subObj->clone());

    evaluateCudaNodes(nodes, X1, "subtract");
    
    std::unique_ptr<Node> mulObj = std::unique_ptr<Node>(new NodeMultiply());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(mulObj->clone());

    evaluateCudaNodes(nodes, X1, "multiply");
    
    std::unique_ptr<Node> divObj = std::unique_ptr<Node>(new NodeDivide());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(divObj->clone());

    evaluateCudaNodes(nodes, X1, "divide");
    
    std::unique_ptr<Node> expObj = std::unique_ptr<Node>(new NodeExponent());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(expObj->clone());

    evaluateCudaNodes(nodes, X1, "exponent");
    
    MatrixXd X2(1,4); 
    X2 << 0.0, 1.0, 2.0, 3.0;
    
    std::unique_ptr<Node> cosObj = std::unique_ptr<Node>(new NodeCos());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cosObj->clone());

    evaluateCudaNodes(nodes, X2, "cos");
    
    std::unique_ptr<Node> cubeObj = std::unique_ptr<Node>(new NodeCube());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cubeObj->clone());

    evaluateCudaNodes(nodes, X2, "cube");
    
    std::unique_ptr<Node> exptObj = std::unique_ptr<Node>(new NodeExponential());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(exptObj->clone());

    evaluateCudaNodes(nodes, X2, "exponential");
    
    std::unique_ptr<Node> gaussObj = std::unique_ptr<Node>(new NodeGaussian());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(gaussObj->clone());

    evaluateCudaNodes(nodes, X2, "gaussian");
    
    std::unique_ptr<Node> logObj = std::unique_ptr<Node>(new NodeLog());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logObj->clone());

    evaluateCudaNodes(nodes, X2, "log");
    
    std::unique_ptr<Node> logitObj = std::unique_ptr<Node>(new NodeLogit());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logitObj->clone());

    evaluateCudaNodes(nodes, X2, "logit");
    
    std::unique_ptr<Node> signObj = std::unique_ptr<Node>(new NodeSign());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(signObj->clone());

    evaluateCudaNodes(nodes, X2, "sign");
    
    std::unique_ptr<Node> sinObj = std::unique_ptr<Node>(new NodeSin());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sinObj->clone());

    evaluateCudaNodes(nodes, X2, "sin");
    
    std::unique_ptr<Node> sqrtObj = std::unique_ptr<Node>(new NodeSqrt());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sqrtObj->clone());

    evaluateCudaNodes(nodes, X2, "sqrt");
    
    std::unique_ptr<Node> squareObj = std::unique_ptr<Node>(new NodeSquare());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(squareObj->clone());

    evaluateCudaNodes(nodes, X2, "square");
    
    std::unique_ptr<Node> stepObj = std::unique_ptr<Node>(new NodeStep());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(stepObj->clone());

    evaluateCudaNodes(nodes, X2, "step");
    
    std::unique_ptr<Node> tanObj = std::unique_ptr<Node>(new NodeTanh());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(tanObj->clone());

    evaluateCudaNodes(nodes, X2, "tan");
    
    MatrixXd X3(2,3); 
    X3 << 0.0, 1.0, 1.0,
          0.0, 1.0, 0.0;
          
    std::unique_ptr<Node> b1 = std::unique_ptr<Node>(new NodeVariable<bool>(0, 'b'));
    std::unique_ptr<Node> b2 = std::unique_ptr<Node>(new NodeVariable<bool>(1, 'b'));
          
    std::unique_ptr<Node> andObj = std::unique_ptr<Node>(new NodeAnd());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(andObj->clone());

    evaluateCudaNodes(nodes, X3, "and");
    
    std::unique_ptr<Node> orObj = std::unique_ptr<Node>(new NodeOr());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(orObj->clone());

    evaluateCudaNodes(nodes, X3, "or");
    
    std::unique_ptr<Node> xorObj = std::unique_ptr<Node>(new NodeXor());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(xorObj->clone());

    evaluateCudaNodes(nodes, X3, "xor");
    
    MatrixXd X4(2,3); 
    X4 << 1.0, 2.0, 3.0,
          1.0, 1.0, 4.0;
    
    std::unique_ptr<Node> eqObj = std::unique_ptr<Node>(new NodeEqual());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(eqObj->clone());

    evaluateCudaNodes(nodes, X4, "equal");
    
    std::unique_ptr<Node> geqObj = std::unique_ptr<Node>(new NodeGEQ());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(geqObj->clone());

    evaluateCudaNodes(nodes, X4, "GEQ");
    
    std::unique_ptr<Node> gtObj = std::unique_ptr<Node>(new NodeGreaterThan());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(gtObj->clone());

    evaluateCudaNodes(nodes, X4, "GreaterThan");
    
    std::unique_ptr<Node> leqObj = std::unique_ptr<Node>(new NodeLEQ());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(leqObj->clone());

    evaluateCudaNodes(nodes, X4, "LEQ");
    
    std::unique_ptr<Node> ltObj = std::unique_ptr<Node>(new NodeLessThan());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(ltObj->clone());

    evaluateCudaNodes(nodes, X4, "LessThan");
    
    MatrixXd X5(2,3); 
    X5 << 1.0, 2.0, 3.0,
          1.0, 0.0, 1.0;
          
    std::unique_ptr<Node> ifObj = std::unique_ptr<Node>(new NodeIf());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(ifObj->clone());

    evaluateCudaNodes(nodes, X5, "If");
    
    MatrixXd X6(3,3); 
    X6 << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          0.0, 1.0, 0.0;
          
    std::unique_ptr<Node> b3 = std::unique_ptr<Node>(new NodeVariable<bool>(2, 'b'));
          
    std::unique_ptr<Node> iteObj = std::unique_ptr<Node>(new NodeIfThenElse());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(b3->clone());
    nodes.push_back(iteObj->clone());

    evaluateCudaNodes(nodes, X6, "IfThenElse");
    
    MatrixXd X7(1,2); 
    X7 << 1.0, 0.0;
    
    std::unique_ptr<Node> notObj = std::unique_ptr<Node>(new NodeNot());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(notObj->clone());

    evaluateCudaNodes(nodes, X7, "Not");
}
#endif

