#include "testsHeader.h"

/**
Notes
Add import from util for 2d Gauss
**/

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

ArrayXd limited(ArrayXd x)
{
    //cout << "\n From limited function\n INPUT \n"<<x<<"\n";
    x = (isnan(x)).select(0,x);
    x = (x < MIN_DBL).select(MIN_DBL,x);
    x = (x > MAX_DBL).select(MAX_DBL,x);
    
    //cout << "\n From limited function\n OUTPUT \n"<<x<<"\n";
    return x;
};

/*ArrayXd evaluateProgram(NodeVector& program, MatrixXd data, VectorXd labels)

	FT::Stacks stack;

	std::cout << "Running evaluation.\n";
	// Iterate through program and calculate results 
	for (const auto& n : program) {
		std::cout << "Running: " << n->name << "\n";
		n->evaluate(data, labels, z, stack);
		std::cout << "result:" << stack_f[stack_f.size() - 1] << "\n--";
	}

	std::cout << "Returning result.\n" << stack_f[stack_f.size() - 1] << "\n-----------------\n";
	return pop(stack_f);
}*/

Node* parseToNode(std::string token) {
	if (token == "+") {
    	return new FT::NodeAdd();
    } else if (token == "-") {
    	return new FT::NodeSubtract();
    } else if (token == "/") {
    	return new FT::NodeDivide();
    } else if (token == "*") {
    	return new FT::NodeMultiply();
    } else if (token == "cos") {
    	return new FT::NodeCos();
    } else if (token == "sin") {
    	return new FT::NodeSin();
    } else if (token == "x0") {
    	return new FT::NodeVariable(0);
    } else if (token == "x1") {
    	return new FT::NodeVariable(1);
    } else if (token == "exponent") {
    	return new FT::NodeExponent();
    } else if (token == "max") {
    	return new FT::NodeMax();
    } else if (token == "xor") {
    	return new FT::NodeXor();
    } else if (token == "step") {
    	return new FT::NodeStep();
    }
}

FT::NodeVector programGen() {
	FT::NodeVector program;
	std::string txt = "x0 x1 cos";

	char ch = ' ';
	size_t pos = txt.find( ch );
    size_t initialPos = 0;

    // Decompose statement
    std::string token;
    while( pos != std::string::npos ) {
    	token = txt.substr( initialPos, pos - initialPos );

        program.push_back(unique_ptr<Node>(parseToNode(token)));

        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    token = txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 );
    program.push_back(unique_ptr<Node>(parseToNode(token)));

    return program;
}

void testDummyProgram(FT::NodeVector p0, int iters) {
	
	std::cout << "Testing program: [";
	
	for (const auto& n : p0) {
		std::cout << n->name << ", ";
	}
	std::cout << "]\n";

	// Create input data and labels
	MatrixXd x(2, 2);
	VectorXd y(2);
	x << 7.3, 6.7, 
		 12.4, 13.2;

	y << 9.0, 
		 8.0;

    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z; 
	// Params
	double learning_rate = 0.1;
    int bs = 1; 
    FT::Individual ind;
    ind.program = p0;
    FT::Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  2,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  0.5,                              //cross_rate
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5,                             // feedback 
                      "mse",                           //scoring function
                      "",                               // feature names
                      true,                             // backprop
                      iters,                            // iterations
                      learning_rate,
                      bs,
                      false,
                      -1,
                      false);
                      
    Data data(x, y, Z);
    
	// AutoBackProp(PROGRAM, COST_FUNCTION, D_COST_FUNCTION, INPUT_DATA, LABELS, ITERS, LEARNING RATE);
	FT::AutoBackProp* engine = new FT::AutoBackProp("mse", iters, learning_rate);
    engine->run(ind, data, params); // Update pointer to NodeVector internally

    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

	// Make sure internal NodeVector updated
}

TEST(BackProp, DerivativeTest)
{
	Trace trace;
	//vector<ArrayXd> inputs;
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
	trace.f.push_back(input1);
	trace.f.push_back(input2);

	// ADD NODE CHECK -------------------------------------------------------------------------------
	NodeDx* toTest = new FT::NodeAdd();
	
	// Derivative wrt to first input
	ArrayXd expectedDerivative(5, 1);
	expectedDerivative(0,0) = toTest->W[0];
	expectedDerivative(1,0) = toTest->W[0];
	expectedDerivative(2,0) = toTest->W[0];
	expectedDerivative(3,0) = toTest->W[0];
	expectedDerivative(4,0) = toTest->W[0];
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
		
	expectedDerivative(0,0) = toTest->W[1];
	expectedDerivative(1,0) = toTest->W[1];
	expectedDerivative(2,0) = toTest->W[1];
	expectedDerivative(3,0) = toTest->W[1];
	expectedDerivative(4,0) = toTest->W[1];

	// Derivative wrt to second input
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// Derivative wrt to first weight
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);	
    
	// Derivative wrt to second weight
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
    

	// SUB NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSubtract({1,1});
    
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -1;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -1;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
		
    expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;

    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
    
    expectedDerivative(0,0) = -0;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -2;
	expectedDerivative(3,0) = -3;
	expectedDerivative(4,0) = -4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);

	// MULT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeMultiply({1,1});
	
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 4;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);
    
    // DIV NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeDivide({1,1});
	
	expectedDerivative(0,0) = MAX_DBL;	// Div by 0 (limited to 0)
	expectedDerivative(1,0) = 1.0/1;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 1.0/4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	 
	expectedDerivative(0,0) = MIN_DBL;	// Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/4;
	expectedDerivative(3,0) = -1.0/9;
	expectedDerivative(4,0) = -0.0/16; 
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = MAX_DBL;	// Div by 0
	expectedDerivative(1,0) = 3.0/1;
	expectedDerivative(2,0) = 2.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 0.0/4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -MAX_DBL;	//Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/2;
	expectedDerivative(3,0) = -1.0/3;
	expectedDerivative(4,0) = -0.0/4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);

	// x^y NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponent({1.0,1.0});
	
	expectedDerivative(0,0) = 0 * pow(4,0)/4; 
	expectedDerivative(1,0) = 1 * pow(3,1)/3;
	expectedDerivative(2,0) = 2 * pow(2,2)/2;
	expectedDerivative(3,0) = 3 * pow(1,3)/1;
	expectedDerivative(4,0) = 4 * pow(0,4)/0; // div by 0
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 1 * pow(4,0) * log(4); 
    expectedDerivative(1,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(2,0) = 1 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(1,3) * log(1);
	expectedDerivative(4,0) = 1 * pow(0,4) * log(0); // log 0
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 0 * pow(4,0)/1;
	expectedDerivative(1,0) = 1 * pow(3,1)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/1;
	expectedDerivative(3,0) = 3 * pow(1,3)/1;
	expectedDerivative(4,0) = 4 * pow(0,4)/1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
	expectedDerivative(4,0) = 4 * pow(0,4) * log(0); // Log by 0
	expectedDerivative(3,0) = 3 * pow(1,3) * log(1);
	expectedDerivative(2,0) = 2 * pow(2,2) * log(2);
	expectedDerivative(1,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(0,0) = 0 * pow(4,0) * log(4);
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);
	
	// COS NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeCos({1.0});
	
	expectedDerivative(0,0) = -1 * sin(4);
	expectedDerivative(1,0) = -1 * sin(3);
	expectedDerivative(2,0) = -1 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -1 * sin(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -4 * sin(4);
	expectedDerivative(1,0) = -3 * sin(3);
	expectedDerivative(2,0) = -2 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -0 * sin(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// SIN NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSin({1.0});
	
	expectedDerivative(0,0) = 1 * cos(4);
	expectedDerivative(1,0) = 1 * cos(3);
	expectedDerivative(2,0) = 1 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 1 * cos(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 4 * cos(4);
	expectedDerivative(1,0) = 3 * cos(3);
	expectedDerivative(2,0) = 2 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 0 * cos(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// ^3 NODE CHECK  -------------------------------------------------------------------------------
	toTest = new FT::NodeCube({1.0});
	
	expectedDerivative(0,0) = 3 * pow(4,2);
	expectedDerivative(1,0) = 3 * pow(3,2);
	expectedDerivative(2,0) = 3 * pow(2,2);
	expectedDerivative(3,0) = 3 * pow(1,2);
	expectedDerivative(4,0) = 3 * pow(0,2);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 3 * 64;
	expectedDerivative(1,0) = 3 * 27;
	expectedDerivative(2,0) = 3 *  8;
	expectedDerivative(3,0) = 3 *  1;
	expectedDerivative(4,0) = 3 *  0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// e^x NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponential({1.0});
	
	expectedDerivative(0,0) = 1 * exp(4);
	expectedDerivative(1,0) = 1 * exp(3);
	expectedDerivative(2,0) = 1 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 1 * exp(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);

	expectedDerivative(0,0) = 4 * exp(4);
	expectedDerivative(1,0) = 3 * exp(3);
	expectedDerivative(2,0) = 2 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 0 * exp(0);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// GAUS NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeGaussian({1.0});
	
	expectedDerivative(0,0) = 2 * (1 - 4) * exp(-pow(1 - 4, 2));
	expectedDerivative(1,0) = 2 * (1 - 3) * exp(-pow(1 - 3, 2));
	expectedDerivative(2,0) = 2 * (1 - 2) * exp(-pow(1 - 2, 2));
	expectedDerivative(3,0) = 2 * (1 - 1) * exp(-pow(1 - 1, 2));
	expectedDerivative(4,0) = 2 * (1 - 0) * exp(-pow(1 - 0, 2));
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 2 * (4 - 1) * exp(-pow(1 - 4, 2));
	expectedDerivative(1,0) = 2 * (3 - 1) * exp(-pow(1 - 3, 2));
	expectedDerivative(2,0) = 2 * (2 - 1) * exp(-pow(1 - 2, 2));
	expectedDerivative(3,0) = 2 * (1 - 1) * exp(-pow(1 - 1, 2));
	expectedDerivative(4,0) = 2 * (0 - 1) * exp(-pow(1 - 0, 2));
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// LOG NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeLog({1.0});
	
	expectedDerivative(0,0) = 1.0/4;
	expectedDerivative(1,0) = 1.0/3;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = MAX_DBL; // Check if this is intended
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// LOGIT NODE CHECK------------------------------------------------------------------------------
	toTest = new FT::NodeLogit({1.0});
	
	expectedDerivative(0,0) = (1 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (1 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (1 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (1 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
    
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = (4 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (3 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (2 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (0 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
    
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// RELU NODE CHECK------------------------------------------------------------------------------
    /*toTest = new FT::NodeRelu({1.0});
    
    expectedDerivative(0,0) = 1;
    expectedDerivative(1,0) = 1;
    expectedDerivative(2,0) = 1;
    expectedDerivative(3,0) = 1;
    expectedDerivative(4,0) = 0.01;
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
    expectedDerivative(0,0) = 4;
    expectedDerivative(1,0) = 3;
    expectedDerivative(2,0) = 2;
    expectedDerivative(3,0) = 1;
    expectedDerivative(4,0) = 0.01;
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);*/

	// SQRT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeSqrt({1.0});
	
	expectedDerivative(0,0) = 1/(2 * sqrt(4));
	expectedDerivative(1,0) = 1/(2 * sqrt(3));
	expectedDerivative(2,0) = 1/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 1/(2 * sqrt(0)); // divide by zero
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 4/(2 * sqrt(4));
	expectedDerivative(1,0) = 3/(2 * sqrt(3));
	expectedDerivative(2,0) = 2/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 0/(2 * sqrt(0)); //divide by zero
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// ^2  NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSquare({1.0});
	
	expectedDerivative(0,0) = 2 * 1 * 4;
	expectedDerivative(1,0) = 2 * 1 * 3;
	expectedDerivative(2,0) = 2 * 1 * 2;
	expectedDerivative(3,0) = 2 * 1 * 1;
	expectedDerivative(4,0) = 2 * 1 * 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 2 * 16;
	expectedDerivative(1,0) = 2 *  9;
	expectedDerivative(2,0) = 2 *  4;
	expectedDerivative(3,0) = 2 *  1;
	expectedDerivative(4,0) = 2 *  0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// TANH NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeTanh({1.0});
	
	expectedDerivative(0,0) = 0.0013409506830258968799702;
	expectedDerivative(1,0) = 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 4 * 0.0013409506830258968799702;
	expectedDerivative(1,0) = 3 * 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 2 * 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 1 * 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
}


/*TEST(BackProp, PropogationTest)
{
    int myNumber = 0;
	string input = "";
	FT::NodeVector program = programGen();
	while (true) {
		//cout << "Please enter number of iterations: ";
	   	//getline(cin, input);
	   	
	   	input = "10";

	   	// This code converts from string to number safely.
	   	stringstream myStream(input);
	   	if (myStream >> myNumber)
	    	break;
	   	cout << "Invalid number, please try again" << endl;
	}
 	cout << "Running with : " << myNumber << endl << endl;
	testDummyProgram(program, myNumber);
}*/
