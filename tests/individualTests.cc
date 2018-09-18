#include "testsHeader.h"

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

TEST(Individual, EvalEquation)
{
	Feat feat(100, 100, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
	
    MatrixXd X(4,2); 
    MatrixXd X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    VectorXd y(4); 
    VectorXd y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;
    
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z; 

    feat.params.init();       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.p_ml = make_shared<ML>(feat.params); // intialize ML
    feat.p_pop = make_shared<Population>(feat.params.pop_size);
    feat.p_eval = make_shared<Evaluation>(feat.params.scorer);

	feat.params.set_terminals(X.rows());
	
	Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv); 
        
    // initial model on raw input
    feat.initial_model(d);
                  
    // initialize population 
    feat.p_pop->init(feat.best_ind, feat.params);
    int i;
    for(i = 0; i < feat.p_pop->individuals.size(); i++){
	    if (!checkBrackets(feat.p_pop->individuals[i].get_eqn()))
            std::cout << "check brackets failed on eqn " << feat.p_pop->individuals[i].get_eqn() << "\n";
        ASSERT_TRUE(checkBrackets(feat.p_pop->individuals[i].get_eqn())); //TODO evaluate if string correct or not
    }
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
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(3)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(4)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.program.subtree(5), 3);
	ASSERT_EQ(a.program.subtree(2), 0);
	ASSERT_EQ(a.program.subtree(1), 1);
	ASSERT_EQ(a.program.subtree(6), 0);
	
	a.program.clear();
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(3)));
	a.program.push_back(std::unique_ptr<Node>(new NodeConstant(true)));
	a.program.push_back(std::unique_ptr<Node>(new NodeIfThenElse()));
	a.program.push_back(std::unique_ptr<Node>(new NodeAdd()));
	
	ASSERT_EQ(a.program.subtree(4), 1);
	ASSERT_EQ(a.program.subtree(5), 0);
	ASSERT_EQ(a.program.subtree(2), 2);
}

TEST(Individual, Complexity)
{
	Individual a;
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(3)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(4)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.complexity(), 14);
	
	a.program.clear();
	a.c = 0;
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.complexity(), 6);
	
	a.program.clear();
	a.c = 0;
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<double>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSin()));
	
	ASSERT_EQ(a.complexity(), 6);
	a.c = 0;
}

