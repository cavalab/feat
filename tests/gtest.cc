#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <shogun/base/init.h>
#include <omp.h>
#include <string>	

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


#include <cstdio>
#include "../src/fewtwo.h"
#include <gtest/gtest.h>

using namespace FT;
 

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
    ASSERT_STREQ("RandomForest", fewtwo.p_ml->type.c_str());
    ASSERT_EQ(sh::EMachineType::CT_BAGGING, fewtwo.p_ml->p_est->get_classifier_type());
    ASSERT_EQ(sh::EProblemType::PT_REGRESSION, fewtwo.p_ml->p_est->get_machine_problem_type());
    
    fewtwo.set_classification(true);
    ASSERT_EQ(sh::EProblemType::PT_MULTICLASS, fewtwo.p_ml->p_est->get_machine_problem_type());
    
    fewtwo.set_verbosity(2);
    ASSERT_EQ(2, fewtwo.params.verbosity);
    
    fewtwo.set_max_stall(2);
    ASSERT_EQ(2, fewtwo.params.max_stall);
    
    fewtwo.set_selection("pareto");
    ASSERT_STREQ("pareto", fewtwo.p_sel->get_type().c_str());
    
    fewtwo.set_survival("lexicase");
    ASSERT_STREQ("lexicase", fewtwo.p_surv->get_type().c_str());
    
    fewtwo.set_cross_rate(0.6);
    EXPECT_EQ(0.6, fewtwo.p_variation->get_cross_rate());
    
    fewtwo.set_otype('b');
    ASSERT_EQ('b', fewtwo.params.otype);
    
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
    fewtwo.p_pop->init(fewtwo.params);
    int i;
    for(i = 0; i < fewtwo.p_pop->individuals.size(); i++)
	    EXPECT_STREQ("", fewtwo.p_pop->individuals[i].get_eqn().c_str()); //TODO evaluate if string correct or not
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
	stack_b.push_back(Z1);
	stack_b.push_back(Z2);
	
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
	
	//TODO for NodeIf, NodeIfThenElse, NodeVariable, NodeConstant(both types)
} 

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
