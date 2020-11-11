#include "testsHeader.h"

TEST(Feat, SettingFunctions)
{
	Feat feat(100, 10, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
    
    feat.set_pop_size(200);
    ASSERT_EQ(200, feat.get_pop_size());
    
    feat.set_gens(50);
    ASSERT_EQ(50, feat.get_gens());
    
    feat.set_ml("RandomForest");
    ASSERT_STREQ("RandomForest", feat.params.ml.c_str());
    
    feat.set_verbosity(3);
    ASSERT_EQ(3, feat.params.verbosity);
    
    feat.set_max_stall(2);
    ASSERT_EQ(2, feat.params.max_stall);
    
    feat.set_selection("nsga2");
    ASSERT_STREQ("nsga2", feat.selector.get_type().c_str());
    
    feat.set_survival("lexicase");
    ASSERT_STREQ("lexicase", feat.survivor.get_type().c_str());
    
    feat.set_cross_rate(0.6);
    EXPECT_EQ(6, (int)(feat.variator.get_cross_rate()*10));
    
    feat.set_otype('b');
    ASSERT_EQ('b', feat.params.otypes[0]);
    
    feat.set_verbosity(1);
    feat.set_functions("+,-");
    ASSERT_EQ(2, feat.params.functions.size());
    ASSERT_STREQ("+", feat.params.functions[0]->name.c_str());
    ASSERT_STREQ("-", feat.params.functions[1]->name.c_str());
    
    feat.set_max_depth(2);
    ASSERT_EQ(2, feat.params.max_depth);

    feat.set_max_dim(15);
    ASSERT_EQ(15, feat.params.max_dim);
    
    feat.set_erc(false);
    ASSERT_EQ(false, feat.params.erc);
    
    feat.set_random_state(2);
    //TODO test random state seed
}


TEST(Feat, predict)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);

    
    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    feat.set_n_jobs(1);
    /* cout << "line 143: predict\n"; */
    feat.fit(X, y);
    /* cout << "line 145: done with fit\n"; */
    
    X << 0,1,  
         0.4,0.8,  
         0.8,  0.,
         0.9,  0.0,
         0.9, -0.4,
         0.5, -0.8,
         0.1,-0.9;
    
    ASSERT_EQ(feat.predict(X).size(), 7);      
}

TEST(Feat, transform)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    
    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    feat.fit(X, y);
    
    X << 0,1,  
         0.4,0.8,  
         0.8,  0.,
         0.9,  0.0,
         0.9, -0.4,
         0.5, -0.8,
         0.1,-0.9;
       
    MatrixXf res = feat.transform(X);
    ASSERT_EQ(res.cols(), 7);

    if (res.rows() > feat.params.max_dim){
        std::cout << "res.rows(): " << res.rows() << 
                    ", res.cols(): " << res.cols() << 
                    ", params.max_dim: " << feat.params.max_dim << "\n";
    }
    ASSERT_TRUE(res.rows() <= feat.params.max_dim);
}

TEST(Feat, fit_predict)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    
    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    feat.set_verbosity(1);
         
    ASSERT_EQ(feat.fit_predict(X, y).size(), 7);
}

TEST(Feat, fit_transform)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
   
    feat.set_verbosity(1);

    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    
    MatrixXf res = feat.fit_transform(X, y);
    ASSERT_EQ(res.cols(), 7);
    
    if (res.rows() > feat.params.max_dim)
        std::cout << "res.rows(): " << res.rows() << 
                    ", res.cols(): " << res.cols() << 
                    ", params.max_dim: " << feat.params.max_dim << "\n";

    ASSERT_TRUE(res.rows() <= feat.params.max_dim);
}

TEST(Feat, simplification)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    feat.set_verbosity(2);

    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z;
    DataRef d(X, y, Z, false);

    // test NOT(NOT(NOT(x<t)))
    cout << "\n\n test NOT(NOT(NOT(x<t)))\n";
    Individual test_ind; 
	test_ind.program.push_back(
            std::unique_ptr<Node>( new NodeVariable<float>(0)));
	test_ind.program.push_back(
            std::unique_ptr<Node>( new NodeSplit<float>()));
	test_ind.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind.program.push_back(std::unique_ptr<Node>(new NodeNot()));

    bool pass = true;
    test_ind.fit(*d.o, feat.params, pass);
    feat.simplify_model(d, test_ind);

    ASSERT_EQ(test_ind.program.size(), 3);
    
    // test [NOT(NOT(NOT(x1<t)))][NOT(NOT(NOT(NOT(x0<t))))]
    cout << "\n\ntest [NOT(NOT(NOT(x1<t)))][NOT(NOT(NOT(NOT(x0<t))))] " << endl;
    feat.set_verbosity(2);

    Individual test_ind2; 
	test_ind2.program.push_back(
            std::unique_ptr<Node>( new NodeVariable<float>(0)));
	test_ind2.program.push_back(
            std::unique_ptr<Node>( new NodeSplit<float>()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind2.program.push_back(std::unique_ptr<Node>(new NodeNot()));
	test_ind2.program.push_back(
            std::unique_ptr<Node>( new NodeVariable<float>(1)));
	test_ind2.program.push_back(
            std::unique_ptr<Node>( new NodeSplit<float>()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));
	test_ind2.program.push_back(std::unique_ptr<Node>( new NodeNot()));

    pass = true;
    test_ind2.fit(*d.o, feat.params, pass);
    feat.simplify_model(d, test_ind2);

    ASSERT_EQ(test_ind2.program.size(), 5);

    cout << "\n\ntest repeat feature: [(x0<t)][(x0<t)]" << endl;
    Individual test_ind3; 
	test_ind3.program.push_back(
            std::unique_ptr<Node>( new NodeVariable<float>(0)));
	test_ind3.program.push_back(
            std::unique_ptr<Node>( new NodeSplit<float>()));
	test_ind3.program.push_back(
            std::unique_ptr<Node>( new NodeVariable<float>(0)));
	test_ind3.program.push_back(
            std::unique_ptr<Node>( new NodeSplit<float>()));

    pass = true;
    test_ind3.fit(*d.o, feat.params, pass);
    feat.simplify_model(d, test_ind3);

    ASSERT_EQ(test_ind3.program.size(), 2);
}

TEST(Feat, serialization)
{
    Feat feat(100, 10, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    feat.set_verbosity(2);

    MatrixXf X(7,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372,
         0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    
    VectorXf y(7); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
             -1.20648656, -2.68773747;
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z;
    DataRef d(X, y, Z, false);

    // TODO: write this test!
}
