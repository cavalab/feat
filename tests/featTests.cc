#include "testsHeader.h"

TEST(Feat, SettingFunctions)
{
	Feat feat(100, 100, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
    
    feat.set_pop_size(200);
    ASSERT_EQ(200, feat.params.pop_size);
    
    feat.set_generations(50);
    ASSERT_EQ(50, feat.params.gens);
    
    feat.set_ml("RandomForest");
    ASSERT_STREQ("RandomForest", feat.params.ml.c_str());
    //ASSERT_STREQ("RandomForest", feat.p_ml->type.c_str());
    //ASSERT_EQ(sh::EMachineType::CT_BAGGING, feat.p_ml->p_est->get_classifier_type());
    //ASSERT_EQ(sh::EProblemType::PT_REGRESSION, feat.p_ml->p_est->get_machine_problem_type());
    
    //feat.set_classification(true);
    //ASSERT_EQ(sh::EProblemType::PT_MULTICLASS, feat.p_ml->p_est->get_machine_problem_type());
    
    feat.set_verbosity(3);
    ASSERT_EQ(3, feat.params.verbosity);
    
    feat.set_max_stall(2);
    ASSERT_EQ(2, feat.params.max_stall);
    
    feat.set_selection("nsga2");
    ASSERT_STREQ("nsga2", feat.p_sel->get_type().c_str());
    
    feat.set_survival("lexicase");
    ASSERT_STREQ("lexicase", feat.p_surv->get_type().c_str());
    
    feat.set_cross_rate(0.6);
    EXPECT_EQ(6, (int)(feat.p_variation->get_cross_rate()*10));
    
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
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);

    
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
    
    feat.set_n_threads(1);
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
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    
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
    
    feat.fit(X, y);
    
    X << 0,1,  
         0.4,0.8,  
         0.8,  0.,
         0.9,  0.0,
         0.9, -0.4,
         0.5, -0.8,
         0.1,-0.9;
       
    MatrixXd res = feat.transform(X);
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
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    
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
    
    feat.set_verbosity(1);
         
    ASSERT_EQ(feat.fit_predict(X, y).size(), 7);
}

TEST(Feat, fit_transform)
{
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
   
    feat.set_verbosity(1);

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
    
    MatrixXd res = feat.fit_transform(X, y);
    ASSERT_EQ(res.cols(), 7);
    
    if (res.rows() > feat.params.max_dim)
        std::cout << "res.rows(): " << res.rows() << 
                    ", res.cols(): " << res.cols() << 
                    ", params.max_dim: " << feat.params.max_dim << "\n";

    ASSERT_TRUE(res.rows() <= feat.params.max_dim);
}

