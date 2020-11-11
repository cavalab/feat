#include "testsHeader.h"

TEST(Selection, SelectionOperator)
{
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_random_state(666);
    feat.set_scorer("mae");
    
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
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z_t;
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z_v;

	feat.timer.Reset();
	
	/* MatrixXf X_t(X.rows(),int(X.cols()*feat.params.split)); */
    /* MatrixXf X_v(X.rows(),int(X.cols()*(1-feat.params.split))); */
    /* VectorXf y_t(int(y.size()*feat.params.split)), y_v(int(y.size()*(1-feat.params.split))); */
    
    DataRef d(X, y, Z);
    
    d.train_test_split(feat.params.shuffle, feat.params.split);
    
    feat.timer.Reset();
	
	feat.params.init(X, y);       

    feat.set_dtypes(find_dtypes(X));
            
    feat.pop = Population(feat.params.pop_size);    
    feat.evaluator = Evaluation(feat.params.scorer_);

	feat.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    feat.initial_model(d);
                  
    // initialize population 
    feat.pop.init(feat.best_ind, feat.params);
    
    feat.evaluator.fitness(feat.pop.individuals, *d.t, feat.params);
    vector<size_t> parents = feat.selector.select(feat.pop, feat.params, 
            *d.t);
    
    ASSERT_EQ(parents.size(), feat.get_pop_size());
}
