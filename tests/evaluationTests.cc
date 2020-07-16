#include "testsHeader.h"

TEST(Evaluation, mse)
{
	
    Feat ft;
    
    VectorXf yhat(10), y(10), res(10);
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

    // test mean squared error      
    VectorXf loss; 
    float score = mse(y, yhat, loss, ft.params.class_weights);

    if (loss != res)
    {
        std::cout << "loss:" << loss << "\n";
        std::cout << "res:" << res.transpose() << "\n";
    }

    ASSERT_TRUE(loss == res);
    ASSERT_TRUE(score == 28.5);
}

TEST(Evaluation, bal_accuracy)
{
    // test balanced zero one loss
	
    Feat ft;
    
    VectorXf yhat(10), y(10), res(10), loss(10);
	
  
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
           
    float score = bal_zero_one_loss(y, yhat, loss, ft.params.class_weights);
    
    if (loss != res)
    {
        std::cout << "loss:" << loss.transpose() << "\n";
        std::cout << "res:" << res.transpose() << "\n";
    }
    ASSERT_TRUE(loss == res);
    ASSERT_EQ(((int)(score*1000000)), 347222);
   
}

TEST(Evaluation, log_loss)
{
    // test log loss
	
    VectorXf yhat(10), y(10), loss(10);
    //ArrayXXf confidences(10,2);

    // test log loss
    y << 0, 
         0,
         0,
         1,
         1,
         1,
         1,
         1, 
         0,
         0;
   
    //cout << "setting confidences\n";
    yhat << 0.17299064146709298 ,
             0.2848611182312323 ,
             0.4818940550844429 ,
             0.6225638563373588 ,
             0.849437595665999 ,
             0.7758710334865722 ,
             0.6172182787000485 ,
             0.4640091939913109 ,
             0.4396698295617455 ,
             0.47602999293839676 ;
    
    float score = mean_log_loss(y, yhat, loss);
    
    ASSERT_EQ(((int)(score*100000)),45495);
}

TEST(Evaluation, multi_log_loss)
{
    // test log loss   
    VectorXf y(10), loss(10);
    ArrayXXf confidences(10,3);

    // test log loss
    y << 0, 
         0,
         1,
		 2,
		 2,
		 0,
		 1,
		 1,
		 0,
		 2;

  
    //cout << "setting confidences\n";
    confidences << 0.4635044256474209,0.0968001677665267,0.43969540658605233,
					0.6241231423983534,0.04763759878396275,0.3282392588176838,
					0.3204639096468384,0.6057958969556588,0.07374019339750265,
					0.4749388491547049,0.07539667564715603,0.44966447519813907,
					0.3552503205282328,0.019014184065430532,0.6257354954063368,
					0.6492498543925875,0.16946198974704466,0.18128815586036792,
					0.21966902720063683,0.6516907637412063,0.12864020905815685,
					0.02498948061874484,0.7120963741266988,0.26291414525455636,
					0.523429658423623,0.2017439717928997,0.2748263697834774,
					0.2686577572168724,0.449670901690872,0.2816713410922557;
    
    //cout << "running multi_log_loss\n";
	vector<float> weights = {(1-0.4)*3.0, (1-0.4)*3.0, (1-0.3)*3.0};
	/* vector<float> weights; */ 
    /* for (int i = 0; i < y.size(); ++i) */
        /* weights.push_back(1.0); */

    float score = mean_multi_log_loss(y, confidences, loss, weights);
    //cout << "assertion\n";

    ASSERT_EQ(((int)(score*100000)),62344);
}

TEST(Evaluation, fpr)
{
    // test false positive rate  
    VectorXf y(10), loss(10);
    VectorXf yhat(10);

    // test log loss
    y << 0, 
         0,
         1,
		 1,
		 0,
		 0,
		 1,
		 1,
		 0,
		 0;

  
    //cout << "setting yhat\n";
    // three false positives out of 6 negatives = 0.5 false positive rate
    yhat << 1, 
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1;
    //cout << "running multi_log_loss\n";
	vector<float> weights;
	/* vector<float> weights; */ 
    /* for (int i = 0; i < y.size(); ++i) */
        /* weights.push_back(1.0); */

    float score = false_positive_loss(y, yhat, loss, weights);

    ASSERT_EQ(((int)(score*10)),5);

    // 6 false positives out of 6 negatives = 1.0 false positive rate
    yhat << 1, 
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1;
    score = false_positive_loss(y, yhat, loss, weights);

    ASSERT_EQ(((int)(score*10)),10);

    // 0 false positives out of 6 negatives = 0.0 false positive rate
    yhat << 
        0, 
        0,
        0,
		1,
		0,
		0,
		1,
		0,
		0,
		0;

    score = false_positive_loss(y, yhat, loss, weights);

    ASSERT_EQ(((int)(score*10)),0);
}

TEST(Evaluation, fitness)
{
	Feat ft;
	
	MatrixXf X(10,1); 
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
    
    VectorXf y(10); 
    // y = 2*sin(x0) + 3*cos(x0)
    y << 3.0,  3.30384889,  0.57015434, -2.68773747, -3.47453585,
             -1.06686199,  2.32167986,  3.57567996,  1.54221639, -1.90915382;
             
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 
    
    Data d(X, y, z);

    // make population 
    Population pop(2);
    // individual 0 = [sin(x0) cos(x0)]
    pop.individuals[0].program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(0)));
    pop.individuals[0].program.push_back(std::unique_ptr<Node>(new NodeSin({1.0})));
    pop.individuals[0].program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(0)));
    pop.individuals[0].program.push_back(std::unique_ptr<Node>(new NodeCos({1.0})));
    //std::cout << pop.individuals[0].get_eqn() + "\n";
    // individual 1 = [x0] 
    pop.individuals[1].program.push_back(std::unique_ptr<Node>(new NodeVariable<float>(0)));
    //std::cout << pop.individuals[1].get_eqn() + "\n";    

    // get fitness
    Evaluation eval("mse"); 

    eval.fitness(pop.individuals, d, ft.params);
    
    // check results
    //cout << pop.individuals[0].fitness << " , should be near zero\n";
    ASSERT_TRUE(pop.individuals[0].fitness < NEAR_ZERO);
    ASSERT_TRUE(pop.individuals[1].fitness - 60.442868924187906 < NEAR_ZERO);

}

TEST(Evaluation, out_ml)
{
    Feat ft;

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
    
    ft.params.dtypes = find_dtypes(X);
    /* shared_ptr<Evaluation> p_eval = make_shared<Evaluation>(params.scorer); */
    shared_ptr<ML> p_ml = make_shared<ML>();
             
    bool pass = true;
    VectorXf yhat = p_ml->fit_vector(X, y, ft.params, pass);
     
    float mean = ((yhat - y).array().pow(2)).mean();
    
    //cout << "MSE is " << mean;
    
    ASSERT_TRUE(mean < NEAR_ZERO);
}

TEST(Evaluation, marginal_fairness)
{

    Evaluation eval("fpr");
    VectorXf loss(10); 
	MatrixXf X(2,10); 
    X << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
           0,   1,   0,   0,   0,   1,   0,   0,   0,   1;
    /* X << 0.0, */  
    /*      1.0, */  
    /*      2.0, */
    /*      3.0, */
    /*      4.0, */
    /*      5.0, */
    /*      6.0, */
    /*      7.0, */
    /*      8.0, */
    /*      9.0; */

    /* X.transposeInPlace(); */
    
    VectorXf y(10); 
    // y = 2*sin(x0) + 3*cos(x0)
    /* y << ; */
             
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 
    
    Data d(X, y, z, true, {0,1} );

    loss << 0, 1, 0, 1, 0, 1, 0, 1, 0, 0;
    // group 1 total loss: 2/7
    // group 2 total loss: 2/3
    // mean loss: 4/10
    float base_score = loss.mean();
    float score_with_alpha = eval.marginal_fairness(loss, d, base_score, true); 
    float score_no_alpha = eval.marginal_fairness(loss, d, base_score); 
    // fairness with alpha is 
    // 1/10 * 1/2 * (7*|4/10-2/7| + 3*|4/10-2/3|) = 0.08
    // fairness without alpha is 
    // 1/2 * (|4/10-2/7| + |4/10-2/3|) = 0.08
    
    ASSERT_EQ(((int)(score_with_alpha*100)), 8);
    ASSERT_EQ(((int)(score_no_alpha*1000000)), 190476);

}
