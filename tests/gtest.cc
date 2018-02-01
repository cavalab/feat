#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <shogun/base/init.h>
#include <omp.h>
#include <string>
#include <stack>
#include <gtest/gtest.h>	

// stuff being used

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::stoi;
using std::to_string;
using std::stof;
using namespace shogun;

#define private public

#include <cstdio>
#include "../src/fewtwo.h"

using namespace FT;


bool checkBrackets(string str)
{
	stack<char> st;
	int x;
	for(x = 0; x <str.length(); x++)
	{
		if(str[x] == '[' || str[x] == '(')
			st.push(str[x]);
		if(str[x] == ')')
		{
			if(st.top() == '(')
				st.pop();
			else
				return false;
		}
		if(str[x] == ']')
		{
			if(st.top() == '[')
				st.pop();
			else
				return false;
				
			if(!st.empty())
				return false;
		}
	}
	return true;
} 

TEST(Fewtwo, SettingFunctions)
{
	Fewtwo fewtwo(100);
    
    fewtwo.set_pop_size(200);
    ASSERT_EQ(200, fewtwo.params.pop_size);
    ASSERT_EQ(200, fewtwo.p_pop->size());
    
    fewtwo.set_generations(50);
    ASSERT_EQ(50, fewtwo.params.gens);
    
    fewtwo.set_ml("RandomForest");
    ASSERT_STREQ("RandomForest", fewtwo.params.ml.c_str());
    //ASSERT_STREQ("RandomForest", fewtwo.p_ml->type.c_str());
    //ASSERT_EQ(sh::EMachineType::CT_BAGGING, fewtwo.p_ml->p_est->get_classifier_type());
    //ASSERT_EQ(sh::EProblemType::PT_REGRESSION, fewtwo.p_ml->p_est->get_machine_problem_type());
    
    //fewtwo.set_classification(true);
    //ASSERT_EQ(sh::EProblemType::PT_MULTICLASS, fewtwo.p_ml->p_est->get_machine_problem_type());
    
    fewtwo.set_verbosity(2);
    ASSERT_EQ(2, fewtwo.params.verbosity);
    
    fewtwo.set_max_stall(2);
    ASSERT_EQ(2, fewtwo.params.max_stall);
    
    fewtwo.set_selection("pareto");
    ASSERT_STREQ("pareto", fewtwo.p_sel->get_type().c_str());
    
    fewtwo.set_survival("lexicase");
    ASSERT_STREQ("lexicase", fewtwo.p_surv->get_type().c_str());
    
    fewtwo.set_cross_rate(0.6);
    EXPECT_EQ(6, (int)(fewtwo.p_variation->get_cross_rate()*10));
    
    fewtwo.set_otype('b');
    ASSERT_EQ('b', fewtwo.params.otypes[0]);
    
    fewtwo.set_functions("+,-");
    ASSERT_EQ(2, fewtwo.params.functions.size());
    ASSERT_STREQ("+", fewtwo.params.functions[0]->name.c_str());
    ASSERT_STREQ("-", fewtwo.params.functions[1]->name.c_str());
    
    fewtwo.set_max_depth(2);
    ASSERT_EQ(2, fewtwo.params.max_depth);

    fewtwo.set_max_dim(15);
    ASSERT_EQ(15, fewtwo.params.max_dim);
    
    fewtwo.set_erc(false);
    ASSERT_EQ(false, fewtwo.params.erc);
    
    fewtwo.set_random_state(2);
    //TODO test random state seed
}

TEST(Fewtwo, predict)
{
    Fewtwo fewtwo(100);
    
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    fewtwo.set_verbosity(0);
    
    fewtwo.fit(X, y);
    
    X << 0,1,  
         0.4,0.8,  
         0.8,  0.,
         0.9,  0.0,
         0.9, -0.4,
         0.5, -0.8,
         0.1,-0.9;
    
    ASSERT_EQ(fewtwo.predict(X).size(), 7);             //TODO had to remove !bzero ASSERT in set_termincal weights
}

TEST(Fewtwo, transform)
{
    Fewtwo fewtwo(100);
    
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    fewtwo.set_verbosity(0);
    
    fewtwo.fit(X, y);
    
    X << 0,1,  
         0.4,0.8,  
         0.8,  0.,
         0.9,  0.0,
         0.9, -0.4,
         0.5, -0.8,
         0.1,-0.9;
        
    MatrixXd res = fewtwo.transform(X);
    ASSERT_EQ(res.cols(), 7);
    ASSERT_TRUE(res.rows() <= fewtwo.params.max_dim);
}

TEST(Fewtwo, fit_predict)
{
    Fewtwo fewtwo(100);
    
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    fewtwo.set_verbosity(0);
         
    ASSERT_EQ(fewtwo.fit_predict(X, y).size(), 7);
}

TEST(Fewtwo, fit_transform)
{
    Fewtwo fewtwo(100);
    
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    fewtwo.set_verbosity(0);
    
    MatrixXd res = fewtwo.fit_transform(X, y);
    ASSERT_EQ(res.cols(), 7);
    ASSERT_TRUE(res.rows() <= fewtwo.params.max_dim);
}


TEST(Individual, EvalEquation)
{
	Fewtwo fewtwo(100);
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

	fewtwo.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    fewtwo.initial_model(X,y);
                  
    // initialize population 
    fewtwo.p_pop->init(fewtwo.best_ind, fewtwo.params);
    int i;
    for(i = 0; i < fewtwo.p_pop->individuals.size(); i++){
	    if (!checkBrackets(fewtwo.p_pop->individuals[i].get_eqn()))
            std::cout << "check brackets failed on eqn " << fewtwo.p_pop->individuals[i].get_eqn() << "\n";
        ASSERT_TRUE(checkBrackets(fewtwo.p_pop->individuals[i].get_eqn())); //TODO evaluate if string correct or not
    }
}

TEST(NodeTest, Evaluate)
{
	vector<ArrayXd> stack_f;
	vector<ArrayXd> output;
	vector<ArrayXb> stack_b;
	ArrayXd x;
	ArrayXb z;
	
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
    
	std::shared_ptr<Node> addObj = std::shared_ptr<Node>(new NodeAdd());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	addObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> subObj = std::shared_ptr<Node>(new NodeSubtract());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	subObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> mulObj = std::shared_ptr<Node>(new NodeMultiply());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	mulObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> divObj = std::shared_ptr<Node>(new NodeDivide());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	divObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> sqrtObj = std::shared_ptr<Node>(new NodeSqrt());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	sqrtObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> sinObj = std::shared_ptr<Node>(new NodeSin());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	sinObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> cosObj = std::shared_ptr<Node>(new NodeCos());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	cosObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> squareObj = std::shared_ptr<Node>(new NodeSquare());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	squareObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> expObj = std::shared_ptr<Node>(new NodeExponent());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	expObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> exptObj = std::shared_ptr<Node>(new NodeExponential());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	exptObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> logObj = std::shared_ptr<Node>(new NodeLog());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	logObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> andObj = std::shared_ptr<Node>(new NodeAnd());
	
	stack_f.clear();
	stack_b.clear();
	stack_b.push_back(Z1);
	stack_b.push_back(Z2);
	
	andObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> orObj = std::shared_ptr<Node>(new NodeOr());
	
	stack_f.clear();
	stack_b.clear();
	stack_b.push_back(Z1);
	stack_b.push_back(Z2);
	
	orObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> notObj = std::shared_ptr<Node>(new NodeNot());
	
	stack_f.clear();
	stack_b.clear();
	stack_b.push_back(Z1);
	
	notObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> eqObj = std::shared_ptr<Node>(new NodeEqual());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	eqObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	 
	std::shared_ptr<Node> gtObj = std::shared_ptr<Node>(new NodeGreaterThan());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	gtObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> geqObj = std::shared_ptr<Node>(new NodeGEQ());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	geqObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> ltObj = std::shared_ptr<Node>(new NodeLessThan());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	ltObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> leqObj = std::shared_ptr<Node>(new NodeLEQ());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	leqObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> ifObj = std::shared_ptr<Node>(new NodeIf());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_b.push_back(Z1);
	
	ifObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> iteObj = std::shared_ptr<Node>(new NodeIfThenElse());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	stack_b.push_back(Z1);
	
	iteObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::shared_ptr<Node> tanObj = std::shared_ptr<Node>(new NodeTanh());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	tanObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> logitObj = std::shared_ptr<Node>(new NodeLogit());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	logitObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> stepObj = std::shared_ptr<Node>(new NodeStep());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	stepObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> signObj = std::shared_ptr<Node>(new NodeSign());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	signObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	
	std::shared_ptr<Node> xorObj = std::shared_ptr<Node>(new NodeXor());
	
	stack_f.clear();
	stack_b.clear();
	stack_b.push_back(Z1);
	stack_b.push_back(Z2);
	
	xorObj->evaluate(X, Y, stack_f, stack_b);	
	
	z = stack_b.back(); stack_b.pop_back();
	
	ASSERT_FALSE((isinf(z)).any());
	ASSERT_FALSE((isnan(abs(z)).any()));
	
	std::shared_ptr<Node> gausObj = std::shared_ptr<Node>(new NodeGaussian());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	
	gausObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	std::shared_ptr<Node> gaus2dObj = std::shared_ptr<Node>(new Node2dGaussian());
	
	stack_f.clear();
	stack_b.clear();
	stack_f.push_back(X.row(0));
	stack_f.push_back(X.row(1));
	
	gaus2dObj->evaluate(X, Y, stack_f, stack_b);	
	
	x = stack_f.back(); stack_f.pop_back();
	
	ASSERT_FALSE((isinf(x)).any());
	ASSERT_FALSE((isnan(abs(x)).any()));
	
	//TODO NodeVariable, NodeConstant(both types)
}


bool isValidProgram(vector<std::shared_ptr<Node>>& program, unsigned num_features)
{
    //checks whether program fulfills all its arities.
    vector<ArrayXd> stack_f; 
    vector<ArrayXb> stack_b;
    MatrixXd X = MatrixXd::Zero(num_features,2); 
    VectorXd y = VectorXd::Zero(2); 
   
    for (const auto& n : program){
        if ( stack_f.size() >= n->arity['f'] && stack_b.size() >= n->arity['b'])
            n->evaluate(X, y, stack_f, stack_b);
        else
            return false; 
    }
    return true;
}

TEST(Variation, MutationTests)
{
	Fewtwo fewtwo(100);
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

	fewtwo.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    fewtwo.initial_model(X,y);
                  
    // initialize population 
    fewtwo.p_pop->init(fewtwo.best_ind, fewtwo.params);
    
    
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (fewtwo.p_pop->size() < 2*fewtwo.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
		int pass = fewtwo.p_variation->mutate(fewtwo.p_pop->individuals[mom],child,fewtwo.params);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			// give child an open location in F
			child.loc = fewtwo.p_pop->get_open_loc(); 
			//push child into pop
			fewtwo.p_pop->individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < fewtwo.p_pop->individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(fewtwo.p_pop->individuals[i].program, fewtwo.params.terminals.size()));
	
}

TEST(Variation, CrossoverTests)
{
	Fewtwo fewtwo(100);
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

	fewtwo.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    fewtwo.initial_model(X,y);
                  
    // initialize population 
    fewtwo.p_pop->init(fewtwo.best_ind, fewtwo.params);
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (fewtwo.p_pop->size() < 2*fewtwo.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
        int dad = r.random_choice(parents);
        
		int pass = fewtwo.p_variation->cross(fewtwo.p_pop->individuals[mom],fewtwo.p_pop->individuals[dad],child,fewtwo.params);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			// give child an open location in F
			child.loc = fewtwo.p_pop->get_open_loc(); 
			//push child into pop
			fewtwo.p_pop->individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < fewtwo.p_pop->individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(fewtwo.p_pop->individuals[i].program, fewtwo.params.terminals.size()));
}

TEST(Population, PopulationTests)
{
	Fewtwo fewtwo(100);
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

	fewtwo.timer.Reset();
	fewtwo.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    fewtwo.initial_model(X,y);
                  
    // initialize population 
    fewtwo.p_pop->init(fewtwo.best_ind, fewtwo.params);
    
    vector<size_t> current_locs; 
    int i;
    for(i = 0; i < fewtwo.p_pop->individuals.size(); i++)
    {
    	ASSERT_NO_THROW(fewtwo.p_pop->individuals[i]);
	    current_locs.push_back(fewtwo.p_pop->individuals[i].loc);
	}
	    
	size_t loc1 = fewtwo.p_pop->get_open_loc();
	size_t loc2 = fewtwo.p_pop->get_open_loc();
	size_t loc3 = fewtwo.p_pop->get_open_loc();
	size_t loc4 = fewtwo.p_pop->get_open_loc();
	
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc1) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc2) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc3) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc4) != current_locs.end());
	
	//TODO put a check in get_open_loc()
}

TEST(Parameters, ParamsTests)
{                   
	Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  1,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5);                             // feedback
					  
	params.set_max_dim(12);
	ASSERT_EQ(params.max_dim, 12);
	ASSERT_EQ(params.max_size, (pow(2,params.max_depth+1)-1)*params.max_dim);
	
	params.set_max_depth(4);
	ASSERT_EQ(params.max_depth, 4);
	ASSERT_EQ(params.max_size, (pow(2,params.max_depth+1)-1)*params.max_dim);
	
	params.set_terminals(10);
	
	ASSERT_EQ(params.terminals.size(), 10);
	
	vector<double> weights;
	weights.resize(10);
	weights[0] = 10;
	ASSERT_NO_THROW(params.set_term_weights(weights)); //TODO should throw error here as vector is empty just 
	
	params.set_functions("+,-,*,/");
	ASSERT_EQ(params.functions.size(), 4);
	
	int i;
	vector<string> function_names = {"+", "-", "*", "/"};
	for(i = 0; i < 4; i ++)
		ASSERT_STREQ(function_names[i].c_str(), params.functions[i]->name.c_str());
		
	params.set_objectives("fitness,complexity");
	ASSERT_EQ(params.objectives.size(), 2);
	
	ASSERT_STREQ(params.objectives[0].c_str(), "fitness");
	ASSERT_STREQ(params.objectives[1].c_str(), "complexity");
	
	params.set_verbosity(-1);
	ASSERT_EQ(params.verbosity, 1);
	
	params.set_verbosity(10);
	ASSERT_EQ(params.verbosity, 1);
	
	params.set_verbosity(2);
	ASSERT_EQ(params.verbosity, 2);
	
	string str1 = "Hello\n";
	string str2 = params.msg("Hello", 0);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = params.msg("Hello", 1);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = params.msg("Hello", 2);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	params.set_verbosity(1);
	ASSERT_EQ(params.verbosity, 1);
	ASSERT_STREQ("", params.msg("Hello", 2).c_str());
}

TEST(Individual, Check_Dominance)
{
	Individual a, b, c;
	a.obj.push_back(1.0);
	a.obj.push_back(2.0);
	
	b.obj.push_back(1.0);
	b.obj.push_back(3.0);
	
	c.obj.push_back(2.0);
	c.obj.push_back(1.0);
	
	ASSERT_EQ(a.check_dominance(b), 1);			//TODO test fail should be greater than equal to
	ASSERT_EQ(b.check_dominance(a), -1);
	ASSERT_EQ(a.check_dominance(c), 0);
}

TEST(Individual, Subtree)
{
	Individual a;
	
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(1)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(2)));
	a.program.push_back(std::shared_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(3)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(4)));
	a.program.push_back(std::shared_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::shared_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.subtree(5), 3);
	ASSERT_EQ(a.subtree(2), 0);
	ASSERT_EQ(a.subtree(1), 1);
	ASSERT_EQ(a.subtree(6), 0);
	
	a.program.clear();
	
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(1)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(2)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(3)));
	a.program.push_back(std::shared_ptr<Node>(new NodeConstant(true)));
	a.program.push_back(std::shared_ptr<Node>(new NodeIfThenElse()));
	a.program.push_back(std::shared_ptr<Node>(new NodeAdd()));
	
	ASSERT_EQ(a.subtree(4), 1);
	ASSERT_EQ(a.subtree(5), 0);
	ASSERT_EQ(a.subtree(2), 2);
}

TEST(Individual, Complexity)
{
	Individual a;
	
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(1)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(2)));
	a.program.push_back(std::shared_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(3)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(4)));
	a.program.push_back(std::shared_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::shared_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.complexity(), 14);
	
	a.program.clear();
	a.c = 0;
	
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(1)));
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(2)));
	a.program.push_back(std::shared_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.complexity(), 6);
	
	a.program.clear();
	a.c = 0;
	
	a.program.push_back(std::shared_ptr<Node>(new NodeVariable(1)));
	a.program.push_back(std::shared_ptr<Node>(new NodeSin()));
	
	ASSERT_EQ(a.complexity(), 6);
	a.c = 0;
}

TEST(Evaluation, assign_fit)
{
	
	Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  1,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5);                             // feedback
	Individual ind = Individual();
	ind.loc = 0;
	MatrixXd F(10, 1);
	
	VectorXd yhat(10), y(10), res(10);
	yhat << 0.0,
	        1.0,
	        2.0,
	        3.0,
	        4.0, 
	        5.0,
	        6.0,
	        7.0,
	        8.0,
	        9.0;
	        
    y << 0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0;
    
    Evaluation eval = Evaluation();
    
    res << 0.0,
           1.0,
           4.0,
           9.0,
           16.0,
           25.0,
           36.0,
           49.0,
           64.0,
           81.0;
           
    eval.assign_fit(ind, F, yhat, y, params);
    
    ASSERT_TRUE(F.col(ind.loc) == res);
    ASSERT_TRUE(ind.fitness == 28.5);
    
    params.classification = true;
    
    y << 0.0,
         1.0,
         2.0,
         0.0,
         1.0,
         2.0,
         0.0,
         1.0,
         2.0,
         0.0;
         
    res << 0.0,
           0.0,
           0.0,
           1.0,
           1.0,
           1.0,
           1.0,
           1.0,
           1.0,
           1.0;
           
    eval.assign_fit(ind, F, yhat, y, params);
    
    ASSERT_TRUE(F.col(ind.loc) == res);
    ASSERT_TRUE(ind.fitness == 0.7);
    
}

TEST(Evaluation, fitness)
{
	Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  1,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5);                             // feedback                 
	MatrixXd X(10,1); 
    X << 0.0,  
         1.0,  
         2.0,
         3.0,
         4.0,
         5.0,
         6.0,
         7.0,
         8.0,
         9.0;

    X.transposeInPlace();
    
    VectorXd y(10); 
    // y = 2*sin(x0) + 3*cos(x0)
    y << 3.0,  3.30384889,  0.57015434, -2.68773747, -3.47453585,
             -1.06686199,  2.32167986,  3.57567996,  1.54221639, -1.90915382;

    // make population 
    Population pop(2);
    // individual 0 = [sin(x0) cos(x0)]
    pop.individuals[0].program.push_back(std::shared_ptr<Node>(new NodeVariable(0)));
    pop.individuals[0].program.push_back(std::shared_ptr<Node>(new NodeSin()));
    pop.individuals[0].program.push_back(std::shared_ptr<Node>(new NodeVariable(0)));
    pop.individuals[0].program.push_back(std::shared_ptr<Node>(new NodeCos()));
    pop.individuals[0].loc = 0;
    std::cout << pop.individuals[0].get_eqn() + "\n";
    // individual 1 = [x0] 
    pop.individuals[1].program.push_back(std::shared_ptr<Node>(new NodeVariable(0)));
    pop.individuals[1].loc = 1;
    std::cout << pop.individuals[1].get_eqn() + "\n";    
    MatrixXd F(10,2);   // output matrix

    // get fitness
    Evaluation eval; 
    eval.fitness(pop, X, y, F, params);
    // check results
    ASSERT_TRUE(pop.individuals[0].fitness < NEAR_ZERO);
    ASSERT_TRUE(pop.individuals[1].fitness - 60.442868924187906 < NEAR_ZERO);

}

TEST(Evaluation, out_ml)
{
	Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  1,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5);                             // feedback                 
	MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
             
    shared_ptr<Evaluation> p_eval = make_shared<Evaluation>();
    shared_ptr<ML> p_ml = make_shared<ML>(params);
             
    bool pass = true;
    VectorXd yhat = p_ml->out(X, y, params, pass);
    
    double mean = ((yhat - y).array().pow(2)).mean();
    
    cout << "MSE is " << mean;
    
    ASSERT_TRUE(mean < NEAR_ZERO);
}

TEST(Selection, SelectionOperator)
{
    Fewtwo fewtwo(100);
    MatrixXd X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXd y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;

	fewtwo.timer.Reset();
	
	MatrixXd X_t(X.rows(),int(X.cols()*fewtwo.params.split));
    MatrixXd X_v(X.rows(),int(X.cols()*(1-fewtwo.params.split)));
    VectorXd y_t(int(y.size()*fewtwo.params.split)), y_v(int(y.size()*(1-fewtwo.params.split)));
    train_test_split(X,y,X_t,X_v,y_t,y_v,fewtwo.params.shuffle);
        
	fewtwo.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    fewtwo.initial_model(X,y);
                  
    // initialize population 
    fewtwo.p_pop->init(fewtwo.best_ind, fewtwo.params);
    
    fewtwo.F.resize(    X_t.cols(),int(2*fewtwo.params.pop_size));
    fewtwo.p_eval->fitness(*(fewtwo.p_pop), X_t, y_t, fewtwo.F, fewtwo.params);
    vector<size_t> parents = fewtwo.p_sel->select(*(fewtwo.p_pop), fewtwo.F, fewtwo.params);
    
    ASSERT_EQ(parents.size(), fewtwo.get_pop_size());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
