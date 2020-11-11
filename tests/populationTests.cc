#include "testsHeader.h"

TEST(Population, PopulationTests)
{
	Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_n_jobs(1);
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
  
    cout << "dtypes: ";
    for (const auto& d : feat.params.dtypes)
        cout << d << ",";
    cout << "\n";
    feat.pop = Population(feat.params.pop_size);  
    //std::cout << "feat.params.scorer: " << feat.params.scorer << "\n";
    feat.evaluator = Evaluation("mse");

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
    
}

