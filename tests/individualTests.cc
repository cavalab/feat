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
	
    MatrixXf X(4,2); 
    MatrixXf X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    VectorXf y(4); 
    VectorXf y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;
    
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 

    feat.params.init(X, y);       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.pop = Population(feat.params.pop_size);
    feat.evaluator = Evaluation(feat.params.scorer_);

	feat.params.set_terminals(X.rows());
	
	Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv); 
        
    // initial model on raw input
    feat.initial_model(d);
                  
    // initialize population 
    feat.pop.init(feat.best_ind, feat.params);
    int i;
    for(i = 0; i < feat.pop.individuals.size(); i++){
	    if (!checkBrackets(feat.pop.individuals[i].get_eqn()))
            std::cout << "check brackets failed on eqn " << feat.pop.individuals[i].get_eqn() << "\n";
        ASSERT_TRUE(checkBrackets(feat.pop.individuals[i].get_eqn())); //TODO evaluate if string correct or not
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
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(3)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(4)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.program.subtree(5), 3);
	ASSERT_EQ(a.program.subtree(2), 0);
	ASSERT_EQ(a.program.subtree(1), 1);
	ASSERT_EQ(a.program.subtree(6), 0);
	
	a.program.clear();
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(3)));
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
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeAdd()));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(3)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(4)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSubtract()));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.set_complexity(), 14);
	
	a.program.clear();
	a.complexity = 0;
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(2)));
	a.program.push_back(std::unique_ptr<Node>(new NodeMultiply()));
	
	ASSERT_EQ(a.set_complexity(), 6);
	
	a.program.clear();
	a.complexity = 0;
	
	a.program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(1)));
	a.program.push_back(std::unique_ptr<Node>(new NodeSin()));
	
	ASSERT_EQ(a.set_complexity(), 6);
	a.complexity = 0;
}

TEST(Individual, serialization)
{
    /* setup data, then test random individuals. 
     * save the individuals and load them from file. 
     * check that their predictions on train and test match.
     * repeat for different ML pairings.
     */
	
    MatrixXf X(4,2); 
    MatrixXf X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    VectorXf y(4), yb(4); 
    VectorXf y_v(3), yb_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    yb << 1,  0, 1, 1;
    y_v << 0.57015434, -1.20648656, -2.68773747;
    yb_v << 0, 1, 0, 1;
    
    LongData z; 
  

	Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv); 
        
    std::string filename = "test_ind.json";

    for (string model : {"Ridge","Lasso","L1_LR","L2_LR"})
    {
        cout << "model: " << model << endl;
        bool classification = in({"L1_LR","L2_LR"}, model);
        Feat feat(100, 100, "LinearRidgeRegression", classification, 1);
        feat.set_random_state(666);
        if (classification)
        {
            feat.params.init(X,yb);
        }
        else
            feat.params.init(X, y);       

	    feat.params.set_terminals(X.rows());

        for (int i = 0 ; i < 100; ++i)
        {
            Individual ind; 
            cout << "initialize individual\n";
            ind.initialize(feat.params, false, 0);
            bool pass = true;
            cout << "fit individual\n";
            ind.fit(dt, feat.params, pass);
            VectorXf initial_output_train = ind.predict_vector(dt); 
            VectorXf initial_output_test = ind.predict_vector(dv); 
            cout << "saving eqn: " << ind.get_eqn() << endl;
            ind.save(filename); 

            // load the ind and check its output
            Individual loaded_ind;
            loaded_ind.load(filename);
            VectorXf loaded_output_train = loaded_ind.predict_vector(dt);
            VectorXf loaded_output_test = loaded_ind.predict_vector(dv);
            // compare
            /* cout << "fitted eqn: " << ind.get_eqn() << endl; */
            /* cout << "loaded eqn: " << loaded_ind.get_eqn() << endl; */
            ASSERT_EQ(ind.get_eqn(), loaded_ind.get_eqn());

            /* cout << "fitted weights:"; */
            /* for (auto w : ind.w) cout << w << ", "; cout << endl; */
            /* cout << "loaded weights:"; */
            /* for (auto w : loaded_ind.w) cout << w << ", "; cout << endl; */
            ASSERT_EQ(ind.w, loaded_ind.w);

            /* cout << "fitted ML weights:"; */
            /* for (auto w : ind.ml->get_weights()) cout << w << ", "; cout << endl; */
            /* cout << "loaded ml weights:"; */
            /* for (auto w : loaded_ind.ml->get_weights()) cout << w << ", "; cout << endl; */
            ASSERT_EQ(ind.ml->get_weights(), loaded_ind.ml->get_weights());
            /* cout << "initial output train:" << initial_output_train.transpose() */ 
            /*     << endl; */
            /* cout << "loaded output train:" << loaded_output_train.transpose() */ 
            /*     << endl; */

            /* cout << "initial output test:" << initial_output_test.transpose() */ 
            /*     << endl; */
            /* cout << "loaded output test:" << loaded_output_test.transpose() */ 
            /*     << endl; */

            ASSERT_LE((initial_output_train - loaded_output_train).norm(),
                      0.00001);     
            ASSERT_LE((initial_output_test - loaded_output_test).norm(),
                      0.00001);     
        }
    }
    

    //TODO: write this test!
    // train individuals, save behavior
    // save individuals
    // load individuals, predict behavior
    // check saved behavior matches predictions
}


